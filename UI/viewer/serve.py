"""Interactive viser viewer + 4D layout editor for LIFT clips.

Loads (mp4 + camera_da3.npz + layout_track_*.json) and lets the user:
  - inspect per-frame camera poses as tetrahedra (with image thumbnails)
  - edit any keyframe's pose with a 3D transform gizmo
  - record a camera path in the web UI with first-person controls (WASD, space
    up, Ctrl down, arrow keys / mouse to look, F to pin poses shown in-scene), then
    apply trajectory; fine-tune with gizmos
  - draw target 2D bboxes on the LAST frame (add / rename / resize / depth)
  - each bbox -> a 3D AABB candidate (depth-along-ray); scrubbing depth slides the
    candidate forward/backward; box appears in the 3D viewport and is re-projected
    into all 8 keyframe view tiles on the right
  - save edited camera trajectory and bboxes

Usage:
    python serve.py                       # start empty: upload an image or a folder in the browser
    python serve.py --clip <dir>          # pre-load a clip directory (e.g. ../examples/example1)
"""

import argparse
import atexit
import base64
import http.client
import importlib
import io
import json
import os
import random
import shutil
import signal
import struct
import socket
import sys
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import viser
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as R


# ────────────────────────────────────────────────────────────────────────────
# Pillow-backed TrueType rendering — cv2's HERSHEY fonts read as ugly/bold no
# matter how thin we make them. Using a real TrueType through PIL gives us
# proper antialiased glyphs at any size.
# ────────────────────────────────────────────────────────────────────────────

def _candidate_font_paths():
    """A list of likely TrueType locations, in priority order.

    cv2 bundles DejaVu under its Qt resources — that's the most reliable hit
    on a uv/pip venv. Falls back to common Linux/macOS/Windows system paths."""
    paths = []
    try:
        cv2_fonts = os.path.join(os.path.dirname(cv2.__file__), "qt", "fonts")
        if os.path.isdir(cv2_fonts):
            for name in ("DejaVuSans.ttf", "DejaVuSansCondensed.ttf"):
                p = os.path.join(cv2_fonts, name)
                if os.path.isfile(p):
                    paths.append(p)
    except Exception:
        pass
    paths += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:\\Windows\\Fonts\\segoeui.ttf",
    ]
    return paths


_FONT_CACHE = {}

def get_font(size):
    size = max(8, int(round(size)))
    f = _FONT_CACHE.get(size)
    if f is not None:
        return f
    for p in _candidate_font_paths():
        if os.path.isfile(p):
            try:
                f = ImageFont.truetype(p, size)
                _FONT_CACHE[size] = f
                return f
            except Exception:
                continue
    f = ImageFont.load_default()
    _FONT_CACHE[size] = f
    return f


WRAPPER_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LIFT-UI</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<style>
  html,body{margin:0;height:100%;width:100%;max-width:100vw;background:#1d1d22;color:#e6e6e8;
           font-family:system-ui,sans-serif;overflow:hidden;box-sizing:border-box;}
  *,*::before,*::after{box-sizing:inherit;}
  #root{display:flex;height:100dvh;max-height:100dvh;width:100%;max-width:100vw;min-height:0;
        overflow:hidden;}
  /* LEFT column: viser iframe (top) + last-frame editor (bottom). Width fills
     whatever the right column doesn't claim. */
  #left{flex:1 1 auto;min-width:0;min-height:0;display:flex;flex-direction:column;
        overflow:hidden;border-right:1px solid #2a2a2c;}
  /* Each major panel can go browser-fullscreen via the corner control. */
  .panel-with-fs{position:relative;}
  .panel-fs-btn{
    position:absolute;top:6px;right:6px;z-index:60;margin:0;
    padding:5px 10px;font-size:11px;line-height:1.2;cursor:pointer;border-radius:4px;
    background:rgba(24,24,28,.92);color:#e8e8ec;border:1px solid #4a4a52;
    box-shadow:0 1px 4px rgba(0,0,0,.45);font-family:system-ui,sans-serif;
  }
  .panel-fs-btn:hover{background:rgba(40,40,48,.98);border-color:#6a6a78;}
  #viser-wrap{flex:1 1 58%;min-height:min(200px,28vh);overflow:hidden;background:#000;min-width:0;}
  #viser-wrap iframe{width:100%;height:100%;border:0;display:block;}
  #viser-wrap:fullscreen,#viser-wrap:-webkit-full-screen{
    display:flex;flex-direction:column;width:100%;height:100%;min-height:100%;
    background:#000;}
  #viser-wrap:fullscreen iframe,#viser-wrap:-webkit-full-screen iframe{flex:1;min-height:0;border:0;}
  /* Shorter laptop screens: cap editor height so viser keeps room; scroll inside editor. */
  #editor{flex:0 1 auto;height:clamp(180px,min(36vh,360px),440px);min-height:160px;max-height:42vh;
          overflow:auto;padding:8px 8px 10px;
          border-top:1px solid #2a2a2c;background:#0c0c0e;min-width:0;}
  #editor:fullscreen,#editor:-webkit-full-screen{
    display:flex;flex-direction:column;width:100%;
    height:100dvh !important;max-height:100dvh !important;
    overflow:hidden;padding:10px;box-sizing:border-box;}
  #editor:fullscreen #editor-inner,#editor:-webkit-full-screen #editor-inner{
    flex:1;min-height:0;overflow:auto;max-width:100%;width:100%;}
  #editor:fullscreen #editor-canvas-col #canvas-wrap,
  #editor:-webkit-full-screen #editor-canvas-col #canvas-wrap{max-width:100%;}
  #editor:fullscreen #editor-canvas-col,#editor:-webkit-full-screen #editor-canvas-col{
    flex:1.1 1 50%;min-width:min(480px,45vw);}
  #editor:fullscreen #editor-bbox-col,#editor:-webkit-full-screen #editor-bbox-col{
    flex:1 1 280px;min-width:220px;max-width:min(520px,48vw);}
  /* two-column editor: canvas on the left, bbox cards on the right */
  #editor-inner{display:flex;gap:10px;height:100%;width:100%;max-width:100%;margin:0;
                align-items:stretch;min-height:0;}
  #editor-canvas-col{flex:1 1 auto;display:flex;flex-direction:column;gap:6px;
                     min-width:0;max-width:100%;}
  #editor-canvas-col #canvas-wrap{max-width:min(520px,100%);}
  #editor-bbox-col{flex:1 1 300px;min-width:280px;min-height:0;display:flex;flex-direction:column;
                   border-left:1px solid #2a2a2c;padding-left:8px;overflow:hidden;}
  #editor-bbox-col h3{margin-top:0;}
  #bbox-list{flex:1 1 auto;overflow-y:auto;min-height:0;}
  /* RIGHT: keyframe tiles — wrapped so we can collapse to a slim rail */
  #right-rail{display:flex;flex-direction:row;flex:0 0 auto;height:100%;max-height:100dvh;
             min-height:0;background:#08080a;border-left:1px solid #2a2a2c;}
  #right-rail-toggle{
    flex:0 0 24px;width:24px;min-width:24px;align-self:stretch;
    border:none;border-right:1px solid #2a2a2c;cursor:pointer;background:#141418;color:#bbb;
    font-size:13px;line-height:1;padding:0;display:flex;align-items:center;justify-content:center;
    font-family:system-ui,sans-serif;
  }
  #right-rail-toggle:hover{background:#222228;color:#fff;}
  #right-rail.collapsed #right{
    width:0 !important;min-width:0 !important;max-width:0 !important;overflow:hidden !important;
    opacity:0;pointer-events:none;border:none;padding:0;margin:0;
  }
  #right-rail.collapsed #right .panel-fs-btn{display:none;}
  #right-rail.collapsed #tiles{visibility:hidden;height:0;}
  #right{flex:0 0 auto;height:100%;max-height:100dvh;display:flex;flex-direction:column;
         overflow:hidden;background:#08080a;min-height:0;}
  #right:fullscreen,#right:-webkit-full-screen{
    width:100% !important;max-width:100%;height:100%;max-height:100%;
    border-left:0;}
  #right:fullscreen #tiles,#right:-webkit-full-screen #tiles{
    flex:1;min-height:0;height:100%;max-height:none;}
  #tiles{display:grid;
         grid-template-columns:repeat(2, 1fr);
         grid-template-rows:repeat(5, 1fr);
         gap:2px;height:100%;max-height:100dvh;width:100%;
         box-sizing:border-box;min-height:0;}
  .tile{position:relative;background:#1a1a1d;cursor:pointer;
        border:2px solid #2a2a2c;box-sizing:border-box;overflow:hidden;}
  /* Frame 0 = the input image: occupies the full top row but shows the image at
     the SAME size as one keyframe tile, centered (not stretched full-width). */
  .tile.input-tile{grid-column:1 / -1;cursor:default;background:transparent;
                   border:none;overflow:visible;
                   display:flex;align-items:center;justify-content:center;}
  .input-inner{position:relative;height:100%;width:calc((100% - 2px)/2);
               border:2px solid #2a2a2c;box-sizing:border-box;
               background:#1a1a1d;overflow:hidden;}
  .input-inner img{width:100%;height:100%;object-fit:cover;display:block;}
  /* tiles: <img> is just the splat-BG (brightened server-side via ?b=...).
     <canvas.tile-fg> on top is the JS bbox overlay — it's pure matrix
     projection, so bbox edits are instant with no server roundtrip. */
  .tile img{width:100%;height:100%;display:block;object-fit:cover;}
  .tile canvas.tile-fg{position:absolute;top:0;left:0;width:100%;height:100%;
                       pointer-events:none;}
  .tile .cap{position:absolute;top:4px;left:6px;background:rgba(0,0,0,.55);
              padding:1px 6px;font-size:11px;border-radius:3px;color:#fff;
              letter-spacing:.5px;}
  #canvas-wrap{position:relative;display:block;background:#000;line-height:0;
               user-select:none;max-width:100%;}
  /* canvas-bg is brightened server-side via ?b= (same path as tiles), so colours
     stay consistent with tile #8 (target frame). No CSS filter here — otherwise
     it would double-apply on top of the server-side gamma. */
  #canvas-bg{display:block;width:100%;height:auto;}
  #canvas-fg{position:absolute;top:0;left:0;width:100%;height:100%;
             cursor:crosshair;touch-action:none;}
  .row{display:flex;gap:6px;align-items:center;margin:6px 0;font-size:12px;
       flex-wrap:wrap;}
  .swatch{width:14px;height:14px;border-radius:3px;flex:0 0 14px;}
  .bbox-card{border:1px solid #2a2a2c;border-radius:4px;padding:6px;margin:6px 0;
             background:#16161a;}
  .bbox-card.sel{border-color:#ffd84d;}
  .bbox-head{display:flex;gap:6px;align-items:center;font-weight:600;}
  .bbox-head input.name{flex:1;background:#101012;color:#fff;
                        border:1px solid #2a2a2c;padding:3px 6px;}
  .ctrl{display:grid;grid-template-columns:auto 1fr auto;gap:4px 8px;
        align-items:center;font-size:11px;margin-top:6px;}
  .ctrl label{color:#9aa;}
  .ctrl input[type=range]{width:100%;}
  .ctrl input[type=number]{width:64px;background:#101012;color:#fff;
                            border:1px solid #2a2a2c;padding:1px 4px;}
  button{background:#2a2a2c;color:#fff;border:1px solid #3a3a3c;
         padding:5px 9px;cursor:pointer;border-radius:3px;font-size:12px;}
  button.primary{background:#2a5cab;border-color:#4079cf;font-weight:600;}
  button.danger{background:#522;border-color:#733;}
  button:hover{filter:brightness(1.3);}
  .action-row button{padding:7px 12px;font-size:13px;}
  h3{margin:4px 4px;font-size:12px;letter-spacing:.5px;color:#bbb;}
  .meta{font-size:11px;color:#888;padding:0 4px;}
  .hint{color:#888;font-size:11px;margin:4px 2px;}
  /* Narrow laptop / small window: stack bbox list under canvas, drop side border */
  @media (max-width: 920px){
    #editor-inner{flex-direction:column;gap:8px;}
    #editor-bbox-col{border-left:0;padding-left:0;border-top:1px solid #2a2a2c;
                     padding-top:8px;max-height:min(42vh,320px);min-width:0;}
    #editor-canvas-col #canvas-wrap{max-width:100%;}
  }
  @media (max-height: 780px){
    #viser-wrap{min-height:140px;}
    #editor{max-height:38vh;height:clamp(160px,min(34vh,300px),360px);}
  }
  /* Drag-to-resize splitters between panels. */
  .vsplit{flex:0 0 7px;height:7px;min-height:7px;cursor:row-resize;background:#26262b;
          border-top:1px solid #333;border-bottom:1px solid #333;}
  .hsplit{flex:0 0 7px;width:7px;min-width:7px;cursor:col-resize;background:#26262b;
          align-self:stretch;}
  .vsplit:hover,.hsplit:hover{background:#4c6bce;}
  @media (max-width: 920px){ #hsplit-editor{display:none;} }
  .fps-traj-panel{border-top:1px solid #2a2a2c;background:#121218;flex:0 0 auto;
    font-size:12px;padding:6px 10px 10px;color:#ccc;}
  #fps-overlay{display:none;position:fixed;inset:0;z-index:200;background:#000;}
  #fps-overlay.active{display:block;}
  #fps-canvas{display:block;width:100%;height:100%;cursor:crosshair;touch-action:none;}
  #fps-hud{position:absolute;left:8px;right:8px;bottom:8px;background:rgba(12,12,16,.9);
    color:#e8e8ec;padding:10px 12px;border-radius:6px;font-size:12px;line-height:1.5;
    border:1px solid #3a3a44;pointer-events:auto;max-height:38vh;overflow:auto;}
  #fps-hud button{margin:4px 8px 0 0;}
  .modal-overlay{display:none;position:fixed;inset:0;z-index:300;
    background:rgba(0,0,0,.6);align-items:center;justify-content:center;}
  .modal-overlay.active{display:flex;}
  .modal-box{background:#141418;border:1px solid #333;border-radius:10px;
    padding:16px;width:min(640px,92vw);max-height:92vh;overflow:auto;
    box-shadow:0 12px 40px rgba(0,0,0,.6);}
  .modal-box h3{font-size:15px;}
  .gen-field{display:block;font-size:12px;color:#bbb;margin:6px 0;}
  #gen-prompt,#up-prompt{display:block;width:100%;margin-top:4px;background:#0e0e12;
    color:#eee;border:1px solid #333;border-radius:6px;padding:6px;
    font-size:12px;resize:vertical;box-sizing:border-box;}
  #gen-prompt{min-height:160px;line-height:1.4;}
  #up-prompt{min-height:120px;line-height:1.4;}
  .modal-box input[type=number]{background:#0e0e12;color:#eee;
    border:1px solid #333;border-radius:5px;padding:3px 5px;}
  #gen-progress-wrap{display:none;height:8px;background:#0e0e12;border:1px solid #333;
    border-radius:5px;overflow:hidden;margin:8px 0;}
  #gen-progress-bar{height:100%;width:0%;background:#2a5cab;transition:width .2s;}
  #gen-video{display:none;width:100%;margin-top:8px;border-radius:6px;background:#000;}
  #gen-download{color:#6ab0ff;font-size:12px;}
  #up-preview{display:none;max-width:100%;border-radius:6px;margin:6px 0;background:#000;}
</style></head><body>
<div id="root">
  <div id="left">
    <div id="viser-wrap" class="panel-with-fs">
      <button type="button" class="panel-fs-btn" id="fs-viser" title="3D view fullscreen / exit">⛶ 3D</button>
      <iframe id="viser" allow="fullscreen"></iframe>
    </div>
    <div class="fps-traj-panel" id="fps-traj-panel">
      <div class="row" style="margin:0;gap:8px;flex-wrap:wrap;align-items:center;">
        <button type="button" class="primary" id="fps-enter">First-person capture</button>
        <span class="meta">WASD · Space up / Ctrl down · <b>↑↓←→</b>, <b>drag</b> or <b>two-finger scroll</b> to look · double-click to lock pointer · <b>F</b> pin pose · Esc exit</span>
      </div>
      <div class="row" style="margin:8px 0 0;gap:8px;flex-wrap:wrap;align-items:center;">
        <button type="button" class="primary" id="up-open">📤 Upload image / folder</button>
        <button type="button" class="primary" id="gen-open">🎬 Generate video</button>
        <button type="button" id="reset-all" title="Clear the scene, trajectory and boxes">🧹 Reset</button>
        <span class="meta" id="gen-model-status">model: …</span>
      </div>
    </div>
    <div class="vsplit" id="vsplit-editor" title="Drag to resize 3D view / editor"></div>
    <div id="editor" class="panel-with-fs">
      <button type="button" class="panel-fs-btn" id="fs-editor" title="Layout editor fullscreen / exit">⛶ Edit</button>
      <div id="editor-inner">
      <div id="editor-canvas-col">
        <h3>Layout editor — selected keyframe (click a tile on the right to switch frame)</h3>
        <div class="row action-row">
          <button class="primary" onclick="actionAdd()">＋ Add object</button>
          <button onclick="actionRename()">✎ Rename object</button>
          <button class="danger" onclick="actionDelete()">✕ Delete object</button>
          <span class="meta" id="sel-info">no object</span>
        </div>
        <div id="canvas-wrap">
          <img id="canvas-bg" alt="">
          <canvas id="canvas-fg"></canvas>
        </div>
        <div class="hint">
          Canvas + 3D show the <b>selected object</b> only. Selecting an object <b>carries
          its box onto this frame</b> (from its nearest annotated frame) so you can reposition
          it — that's how you set per-frame motion. <b>drag</b> to move/resize · <b>drag empty</b>
          → new box · per-frame (not reprojected); same object across frames shares colour/ID.
        </div>
        <div class="row">
          <button onclick="saveLayout()">💾 Save layout.json</button>
          <button onclick="reloadBg()">↻ Reload bg</button>
          <label class="meta" style="display:flex;gap:6px;align-items:center;">
            ☀ brightness
            <input id="bright" type="range" min="1" max="4" step="0.05" value="1.6"
                   style="width:110px;">
            <span id="bright-val" style="width:30px;">1.60</span>
          </label>
          <label class="meta" style="display:flex;gap:6px;align-items:center;">
            ▭ width
            <input id="bbox-w" type="range" min="1" max="10" step="1" value="3"
                   style="width:90px;">
            <span id="bbox-w-val" style="width:18px;">3</span>
          </label>
          <span class="meta" id="status"></span>
        </div>
      </div>
      <div class="hsplit" id="hsplit-editor" title="Drag to resize canvas / object list"></div>
      <div id="editor-bbox-col">
        <h3>Objects</h3>
        <div class="hint" id="bbox-popover-empty">＋ Add object, then draw on the canvas ←</div>
        <div id="bbox-list"></div>
      </div>
    </div>
    </div>
  </div>
  <div id="right-rail">
    <div class="hsplit" id="hsplit-right" title="Drag to resize keyframe panel"></div>
    <button type="button" id="right-rail-toggle" title="Hide keyframe tiles">◀</button>
    <div id="right" class="panel-with-fs">
    <button type="button" class="panel-fs-btn" id="fs-keyframes" title="Keyframe grid fullscreen / exit">⛶ Keys</button>
    <div id="tiles"></div>
    </div>
  </div>
</div>
<div id="fps-overlay">
  <canvas id="fps-canvas"></canvas>
  <div id="fps-hud">
    <div><b>Violet</b> = first keyframe (fixed) · <b>Orange</b> = F pins · Pinned <b id="fps-cap-count">0</b> · <b>↑↓←→</b> / <b>drag</b> / <b>two-finger scroll</b> to look · <b>double-click</b> locks pointer</div>
    <label class="meta"><input type="radio" name="fps-orient" value="tangent" checked> Forward: along view (interpolate between pins)</label>
    <label class="meta"><input type="radio" name="fps-orient" value="look_at"> Forward: look at scene center</label>
    <div style="margin-top:6px;">
      <button type="button" id="fps-clear-cap">Clear pins</button>
      <button type="button" class="primary" id="fps-apply-cap">Apply trajectory</button>
      <button type="button" id="fps-exit">Exit (Esc)</button>
    </div>
    <div class="meta" id="fps-msg" style="margin-top:6px;"></div>
  </div>
</div>
<div id="up-overlay" class="modal-overlay">
  <div class="modal-box">
    <div class="row" style="justify-content:space-between;align-items:center;margin:0 0 8px;">
      <h3 style="margin:0;">📤 Upload image / folder</h3>
      <button type="button" id="up-close">✕</button>
    </div>
    <div class="hint">Upload an image — MoGe predicts a point cloud + first-frame camera. Then set a first-person camera trajectory, edit the layout, and generate a video in one click.</div>
    <input id="up-file" type="file" accept="image/*" style="margin:8px 0;">
    <div class="hint">…or upload a whole example folder (image + <code>caption.txt</code> + camera <code>.npz</code> + <code>layout_lastframe.json</code>, e.g. <code>examples/example1</code>): the camera path and the last-frame boxes are loaded directly, nothing to draw.</div>
    <input id="up-dir" type="file" webkitdirectory directory multiple style="margin:4px 0 8px;">
    <div class="meta" id="up-dir-info"></div>
    <img id="up-preview" alt="">
    <label class="gen-field">Prompt (optional — scene description)
      <textarea id="up-prompt" rows="5" placeholder="describe the scene…"></textarea>
    </label>
    <div class="row">
      <button type="button" class="primary" id="up-run">Upload &amp; run MoGe</button>
      <span class="meta" id="up-status"></span>
    </div>
  </div>
</div>
<div id="gen-overlay" class="modal-overlay">
  <div class="modal-box">
    <div class="row" style="justify-content:space-between;align-items:center;margin:0 0 8px;">
      <h3 style="margin:0;">🎬 Generate video</h3>
      <button type="button" id="gen-close">✕</button>
    </div>
    <label class="gen-field">Prompt (scene caption)
      <textarea id="gen-prompt" rows="8" placeholder="describe the scene…"></textarea>
    </label>
    <div class="row" style="flex-wrap:wrap;">
      <span class="meta" style="min-width:34px;">seed</span>
      <label class="meta"><input type="radio" name="gen-seed-mode" value="default" checked> <span id="gen-seed-default">42</span></label>
      <label class="meta"><input type="radio" name="gen-seed-mode" value="random"> random</label>
      <label class="meta"><input type="radio" name="gen-seed-mode" value="custom"> custom</label>
      <input id="gen-seed" type="number" value="42" style="width:110px;">
    </div>
    <div class="row" style="flex-wrap:wrap;">
      <label class="meta">steps <input id="gen-steps" type="number" value="50" min="1" max="100" style="width:64px;"></label>
      <label class="meta">cfg <input id="gen-cfg" type="number" value="6.0" step="0.5" min="1" style="width:64px;"></label>
      <span class="meta">81 frames · 640×352 · 16 fps</span>
    </div>
    <div class="row">
      <button type="button" class="primary" id="gen-run">Generate</button>
      <span class="meta" id="gen-status"></span>
    </div>
    <div id="gen-progress-wrap"><div id="gen-progress-bar"></div></div>
    <video id="gen-video" controls playsinline muted loop></video>
    <div class="row" id="gen-result-row" style="display:none;">
      <a id="gen-download" download>⬇ download mp4</a>
      <span class="meta" id="gen-seed-used"></span>
    </div>
  </div>
</div>
<script>
// viser is reverse-proxied through this same origin under /__viser/,
// so SSH-forwarding a single port is enough.
document.getElementById('viser').src = '/__viser/';

// 4 rows fill viewport height; tile column width follows 16:9, but cap so the
// viser + editor column always keeps a usable minimum on laptop-sized windows.
function fitLayout(){
  const w = window.innerWidth;
  const rightEl = document.getElementById('right');
  const rightRail = document.getElementById('right-rail');
  const fs = document.fullscreenElement || document.webkitFullscreenElement;
  if (fs === rightEl) return;
  if (rightRail && rightRail.classList.contains('collapsed')){
    if (rightEl) rightEl.style.width = '0px';
    return;
  }
  // A manual drag on the keyframe splitter pins the width; don't auto-fit over it.
  if (window._manualRightW != null){
    rightEl.style.width = Math.max(200, Math.min(window._manualRightW, w - 300)) + 'px';
    return;
  }
  // Keyframe panel takes ~33% of the window width (leaving the editor + object
  // list the other ~67%), capped so the left column keeps a usable minimum.
  const minLeft = w < 1000 ? 340 : w < 1280 ? 400 : w < 1600 ? 460 : 500;
  const maxRightFromLeft = Math.max(220, w - minLeft);
  const rightW = Math.min(w * 0.33, maxRightFromLeft);
  rightEl.style.width = Math.max(200, rightW) + 'px';
}

(function rightRailCollapse(){
  const rail = document.getElementById('right-rail');
  const btn = document.getElementById('right-rail-toggle');
  if (!rail || !btn) return;
  try {
    if (localStorage.getItem('viewer.rightRailCollapsed') === '1'){
      rail.classList.add('collapsed');
      btn.textContent = '\u25b6';
      btn.title = 'Show keyframe tiles';
    }
  } catch (e) {}
  btn.addEventListener('click', () => {
    rail.classList.toggle('collapsed');
    const c = rail.classList.contains('collapsed');
    btn.textContent = c ? '\u25b6' : '\u25c0';
    btn.title = c ? 'Show keyframe tiles' : 'Hide keyframe tiles';
    try { localStorage.setItem('viewer.rightRailCollapsed', c ? '1' : '0'); } catch (e) {}
    fitLayout();
    try { renderTiles(); drawAllTileOverlays(); } catch (e) {}
    window.dispatchEvent(new Event('resize'));
  });
})();

// ── drag-to-resize splitters (3D↔editor, canvas↔list, keyframe rail width) ──
(function setupResizers(){
  function makeDrag(handle, onMove){
    if (!handle) return;
    handle.addEventListener('pointerdown', e => {
      e.preventDefault();
      try { handle.setPointerCapture(e.pointerId); } catch (_e) {}
      const move = ev => { ev.preventDefault(); onMove(ev); };
      const up = () => {
        try { handle.releasePointerCapture(e.pointerId); } catch (_e) {}
        window.removeEventListener('pointermove', move);
        window.removeEventListener('pointerup', up);
        try { fitCanvas(); renderTiles(); drawAllTileOverlays(); } catch (_e) {}
        window.dispatchEvent(new Event('resize'));
      };
      window.addEventListener('pointermove', move);
      window.addEventListener('pointerup', up);
    });
  }
  // 3D view vs editor (vertical): fix the editor height, viser-wrap flexes to fill.
  const editor = document.getElementById('editor');
  makeDrag(document.getElementById('vsplit-editor'), ev => {
    const box = document.getElementById('left').getBoundingClientRect();
    let hgt = Math.max(140, Math.min(box.bottom - ev.clientY, box.height - 180));
    editor.style.flex = '0 0 ' + hgt + 'px';
    editor.style.height = hgt + 'px';
    editor.style.maxHeight = 'none';
  });
  // Canvas vs object list (horizontal, inside editor-inner).
  const bboxCol = document.getElementById('editor-bbox-col');
  makeDrag(document.getElementById('hsplit-editor'), ev => {
    const inner = document.getElementById('editor-inner').getBoundingClientRect();
    let wid = Math.max(150, Math.min(inner.right - ev.clientX, inner.width - 220));
    bboxCol.style.flex = '0 0 ' + wid + 'px';
    bboxCol.style.maxWidth = 'none';
  });
  // Keyframe rail width (horizontal). Pin via _manualRightW so fitLayout obeys.
  const rightEl = document.getElementById('right');
  makeDrag(document.getElementById('hsplit-right'), ev => {
    let wid = Math.max(200, Math.min(window.innerWidth - ev.clientX - 31,
                                     window.innerWidth - 320));
    window._manualRightW = wid;
    rightEl.style.width = wid + 'px';
    try { renderTiles(); drawAllTileOverlays(); } catch (_e) {}
  });
})();

window.addEventListener('resize', fitLayout);
fitLayout();

function fsActive(){
  return document.fullscreenElement || document.webkitFullscreenElement || null;
}

async function togglePanelFs(el){
  const cur = fsActive();
  try {
    if (cur === el) {
      if (document.exitFullscreen) await document.exitFullscreen();
      else if (document.webkitExitFullscreen) await document.webkitExitFullscreen();
      return;
    }
    if (cur) {
      if (document.exitFullscreen) await document.exitFullscreen();
      else if (document.webkitExitFullscreen) await document.webkitExitFullscreen();
    }
    if (el.requestFullscreen) await el.requestFullscreen({navigationUI:'hide'});
    else if (el.webkitRequestFullscreen) await el.webkitRequestFullscreen();
  } catch (e) { console.warn('fullscreen', e); }
}

function onFullscreenChange(){
  fitLayout();
  window.dispatchEvent(new Event('resize'));
}

document.getElementById('fs-viser').addEventListener('click', () =>
  togglePanelFs(document.getElementById('viser-wrap')));
document.getElementById('fs-editor').addEventListener('click', () =>
  togglePanelFs(document.getElementById('editor')));
document.getElementById('fs-keyframes').addEventListener('click', () =>
  togglePanelFs(document.getElementById('right')));
document.addEventListener('fullscreenchange', onFullscreenChange);
document.addEventListener('webkitfullscreenchange', onFullscreenChange);

const S = {
  W: 1280, H: 720, target_frame: 80,
  // Per-frame layout model (mirror of server state, mutated locally for instant
  // feedback). objects = identity (id/name/colour); boxes = per-frame 2D rects
  // tagged with obj_id + frame. A box shows ONLY on its own frame — no
  // cross-frame reprojection; same obj_id across frames = same object in time.
  objects: [],
  boxes: [],
  currentObjId: null,   // object that newly-drawn boxes get assigned to
  activeFrame: 80,      // video frame the editor is bound to (= active keyframe)
  selectedBoxId: null,  // box selected in the editor canvas (drag/resize)
  drag: null,           // {mode, startX, startY, box?, orig?, cur?}
  bright: 1.6,          // tile/canvas brightness; written by the slider, read in URL params
  bboxWidth: 3,         // bbox outline width in px (canvas + tile); slider in editor bar
  renderVersion: -1,    // bumped server-side when BG cache (or its camera deps) goes stale
  tileW: 0, tileH: 0,
  renderViewIndices: [], // 8 frame indices for the keyframe tiles (in trajectory order)
  pcdPointSize: 0.02,    // world-space point size (Point cloud slider); used by FPS view too
  // Each keyframe gets a unique colour, shared between the 3D frustum and the
  // right-side tile's border. activeKfIdx tracks which one is selected.
  keyframeColors: [],   // 8 of [r, g, b]
  activeKfIdx: 0,
};
const HANDLE = 22;      // hit area (px in image space) for corner handles

// ── per-frame model helpers ────────────────────────────────────────────────
function objById(oid){ return S.objects.find(o => o.id === oid) || null; }
function colorOf(box){ const o = objById(box.obj_id); return o ? o.color : null; }
function objName(box){ const o = objById(box.obj_id); return o ? o.name : '?'; }
function boxesOnFrame(f){ return S.boxes.filter(b => b.frame === f); }
// Editor 2D shows ALL objects placed on the active frame, so adding or
// selecting an object never hides the others already on this frame. The CURRENT
// object is drawn solid + emphasised; the rest are dashed but still visible.
// (Right-side tiles stay unfiltered too.)
function boxesHere(){
  return S.boxes.filter(b => b.frame === S.activeFrame);
}
function boxById(id){ return S.boxes.find(b => b.box_id === id) || null; }

// ── debounced server-sync ────────────────────────────────────────────────
// All edits (drag, slider) update S.boxes locally for an INSTANT redraw, then
// queue a PATCH here. After 500ms of no further input the queue is flushed in
// one go and the server-rendered tiles/canvas refresh once. While anything is
// queued or in-flight, polling is suppressed via busy() so the object-card DOM
// doesn't get torn out from under an in-progress drag.
const pending = new Map();       // bbox_id -> partial fields object
let pendingTimer = null;
let flushBusy = false;
const DEBOUNCE_MS = 500;

function busy(){ return !!(S.drag || pendingTimer || flushBusy || window._fpsBusy || window._switchingKf || window._selectingObj || window._editingName); }

function schedulePatch(id, fields){
  const cur = pending.get(id) || {};
  Object.assign(cur, fields);
  pending.set(id, cur);
  if (pendingTimer) clearTimeout(pendingTimer);
  pendingTimer = setTimeout(flushPending, DEBOUNCE_MS);
}

async function flushPending(){
  pendingTimer = null;
  if (flushBusy) return;
  flushBusy = true;
  try {
    const items = [...pending.entries()];
    pending.clear();
    for (const [id, fields] of items){
      try {
        await fetch('/api/box/' + id, {
          method: 'PATCH',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(fields),
        });
      } catch (e) {}
    }
    // Bboxes don't change the BG — only the JS overlay needs to refresh, and
    // it's already been redrawn synchronously on every input event.
  } finally {
    flushBusy = false;
  }
}

const bg  = document.getElementById('canvas-bg');
const fg  = document.getElementById('canvas-fg');
const ctx = fg.getContext('2d');

function setStatus(s){ document.getElementById('status').textContent = s; }
function reloadBg(){
  bg.src = `/img/canvas_raw.jpg?t=${Date.now()}&b=${S.bright.toFixed(2)}`;
}
function tileUrl(i, t){
  // BG-only tile (no bboxes). The server caches the encoded JPEG by
  // (render_version, brightness), so revisits are ~free. Bboxes are drawn
  // on top via <canvas.tile-fg> — see drawTileOverlay().
  return `/img/tile_bg/${i}.jpg?t=${t}&b=${S.bright.toFixed(2)}`;
}

function rgb(c){ return `rgb(${c[0]},${c[1]},${c[2]})`; }
function rgba(c, a){ return `rgba(${c[0]},${c[1]},${c[2]},${a})`; }
function clamp(v, lo, hi){ return Math.min(Math.max(v, lo), hi); }

function fitCanvas(){
  fg.width  = S.W;
  fg.height = S.H;
  draw();
}

function draw(){
  ctx.clearRect(0, 0, fg.width, fg.height);
  for (const b of boxesHere()){
    const col = colorOf(b); if (!col) continue;
    const sel = (b.box_id === S.selectedBoxId);
    const isCur = (b.obj_id === S.currentObjId);
    const w = b.x2 - b.x1, h = b.y2 - b.y1;
    ctx.fillStyle   = rgba(col, sel ? 0.18 : (isCur ? 0.10 : 0.04));
    ctx.fillRect(b.x1, b.y1, w, h);
    ctx.lineWidth   = isCur ? (S.bboxWidth + (sel ? 2 : 0)) : Math.max(1, S.bboxWidth - 1);
    ctx.setLineDash(isCur ? [] : [7, 5]);   // other objects on this frame: dashed but visible
    ctx.strokeStyle = rgb(col);
    ctx.strokeRect(b.x1, b.y1, w, h);
    ctx.setLineDash([]);

    // label
    const nm = objName(b);
    ctx.font = (isCur ? 'bold ' : '') + '22px system-ui';
    const m = ctx.measureText(nm);
    ctx.fillStyle = rgba([0,0,0], isCur ? 0.6 : 0.4);
    ctx.fillRect(b.x1, b.y1 - 28, m.width + 12, 26);
    ctx.fillStyle = rgb(col);
    ctx.fillText(nm, b.x1 + 6, b.y1 - 8);

    if (sel){
      // corner handles
      const corners = [[b.x1,b.y1],[b.x2,b.y1],[b.x2,b.y2],[b.x1,b.y2]];
      for (const [hx,hy] of corners){
        ctx.fillStyle = '#fff';
        ctx.fillRect(hx-HANDLE/2, hy-HANDLE/2, HANDLE, HANDLE);
        ctx.lineWidth = 2;
        ctx.strokeStyle = rgb(col);
        ctx.strokeRect(hx-HANDLE/2, hy-HANDLE/2, HANDLE, HANDLE);
      }
    }
  }
  // draw temp new bbox during drag
  if (S.drag && S.drag.mode === 'new' && S.drag.cur){
    const d = S.drag;
    ctx.strokeStyle = '#ffd84d';
    ctx.lineWidth = 4;
    ctx.setLineDash([10, 6]);
    const x1 = Math.min(d.startX, d.cur.x), y1 = Math.min(d.startY, d.cur.y);
    const x2 = Math.max(d.startX, d.cur.x), y2 = Math.max(d.startY, d.cur.y);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.setLineDash([]);
  }
}

function evtToImg(e){
  const r = fg.getBoundingClientRect();
  return {
    x: (e.clientX - r.left) / r.width  * S.W,
    y: (e.clientY - r.top ) / r.height * S.H,
  };
}

function hitTest(p){
  const here = boxesHere();
  // top-down iteration
  for (let i = here.length - 1; i >= 0; i--){
    const b = here[i];
    if (b.box_id === S.selectedBoxId){
      const corners = [
        ['resize-tl', b.x1, b.y1],
        ['resize-tr', b.x2, b.y1],
        ['resize-br', b.x2, b.y2],
        ['resize-bl', b.x1, b.y2],
      ];
      for (const [mode, hx, hy] of corners){
        if (Math.abs(p.x - hx) < HANDLE && Math.abs(p.y - hy) < HANDLE)
          return {box: b, mode};
      }
    }
    if (p.x >= b.x1 && p.x <= b.x2 && p.y >= b.y1 && p.y <= b.y2)
      return {box: b, mode: 'move'};
  }
  return null;
}

function cursorFor(mode){
  if (!mode) return 'crosshair';
  if (mode === 'move') return 'move';
  if (mode === 'resize-tl' || mode === 'resize-br') return 'nwse-resize';
  if (mode === 'resize-tr' || mode === 'resize-bl') return 'nesw-resize';
  return 'crosshair';
}

fg.addEventListener('mousemove', e => {
  if (S.drag) return;  // cursor while dragging handled per-mode
  const hit = hitTest(evtToImg(e));
  fg.style.cursor = hit ? cursorFor(hit.mode) : 'crosshair';
});

fg.addEventListener('mousedown', e => {
  if (e.button !== 0) return;
  e.preventDefault();
  const p = evtToImg(e);
  const hit = hitTest(p);
  if (hit){
    S.selectedBoxId = hit.box.box_id;
    // Clicking a box focuses its object (3D view + new-box target follow).
    if (hit.box.obj_id !== S.currentObjId){
      S.currentObjId = hit.box.obj_id;
      fetch('/api/object/' + hit.box.obj_id + '/current', {method: 'POST'}).catch(() => {});
    }
    S.drag = {
      mode: hit.mode,
      box: hit.box,
      startX: p.x, startY: p.y,
      orig: {x1: hit.box.x1, y1: hit.box.y1, x2: hit.box.x2, y2: hit.box.y2},
    };
    fg.style.cursor = cursorFor(hit.mode);
  } else {
    S.selectedBoxId = null;
    S.drag = {mode: 'new', startX: p.x, startY: p.y, cur: p};
  }
  syncList();
  draw();
});

window.addEventListener('mousemove', e => {
  if (!S.drag) return;
  const p = evtToImg(e);
  const d = S.drag;
  if (d.mode === 'new'){
    d.cur = p;
  } else if (d.mode === 'move'){
    const dx = p.x - d.startX, dy = p.y - d.startY;
    const o = d.orig;
    d.box.x1 = clamp(o.x1 + dx, 0, S.W - 1);
    d.box.y1 = clamp(o.y1 + dy, 0, S.H - 1);
    d.box.x2 = clamp(o.x2 + dx, 1, S.W);
    d.box.y2 = clamp(o.y2 + dy, 1, S.H);
  } else {
    // resize
    if (d.mode === 'resize-tl'){ d.box.x1 = clamp(p.x, 0, S.W-1); d.box.y1 = clamp(p.y, 0, S.H-1); }
    if (d.mode === 'resize-tr'){ d.box.x2 = clamp(p.x, 1, S.W);   d.box.y1 = clamp(p.y, 0, S.H-1); }
    if (d.mode === 'resize-br'){ d.box.x2 = clamp(p.x, 1, S.W);   d.box.y2 = clamp(p.y, 1, S.H);   }
    if (d.mode === 'resize-bl'){ d.box.x1 = clamp(p.x, 0, S.W-1); d.box.y2 = clamp(p.y, 1, S.H);   }
    if (d.box.x1 > d.box.x2) [d.box.x1, d.box.x2] = [d.box.x2, d.box.x1];
    if (d.box.y1 > d.box.y2) [d.box.y1, d.box.y2] = [d.box.y2, d.box.y1];
  }
  draw();
  drawAllTileOverlays();
});

window.addEventListener('mouseup', async e => {
  if (!S.drag) return;
  const d = S.drag;
  fg.style.cursor = 'crosshair';
  if (d.mode === 'new'){
    const w = Math.abs(d.cur.x - d.startX), h = Math.abs(d.cur.y - d.startY);
    if (w > 6 && h > 6){
      const x1 = Math.round(Math.min(d.startX, d.cur.x));
      const y1 = Math.round(Math.min(d.startY, d.cur.y));
      const x2 = Math.round(Math.max(d.startX, d.cur.x));
      const y2 = Math.round(Math.max(d.startY, d.cur.y));
      // Boxes belong to the CURRENT object on the ACTIVE frame — auto-create a
      // first object if the user drew before adding one.
      if (S.currentObjId === null){ await addObject(); }
      const r = await fetch('/api/box', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({obj_id: S.currentObjId, frame: S.activeFrame, x1, y1, x2, y2}),
      });
      const nb = await r.json();   // {box_id, obj_id, frame, x1..y2, front_depth, thickness}
      if (nb && nb.box_id !== undefined){
        const ex = boxById(nb.box_id);       // server may return the moved existing box
        if (ex) Object.assign(ex, nb); else S.boxes.push(nb);
        S.selectedBoxId = nb.box_id;
      }
    }
  } else {
    const b = d.box;
    b.x1 = Math.round(b.x1); b.y1 = Math.round(b.y1);
    b.x2 = Math.round(b.x2); b.y2 = Math.round(b.y2);
    schedulePatch(b.box_id, {x1: b.x1, y1: b.y1, x2: b.x2, y2: b.y2});
  }
  S.drag = null;
  syncList();
  draw();
  drawAllTileOverlays();
});

// ── tile grid ────────────────────────────────────────────────────────────
// <img> = splat BG (server, cached by render_version + brightness).
// <canvas class="tile-fg"> = per-frame 2D box overlay (this file). Each tile
// draws only the boxes drawn on its own frame — no 3D projection anymore.

function drawTileOverlay(idx){
  const cvs = document.getElementById('tile-fg-' + idx);
  if (!cvs) return;
  if (cvs.width !== S.tileW || cvs.height !== S.tileH){
    cvs.width = Math.max(1, S.tileW);
    cvs.height = Math.max(1, S.tileH);
  }
  const ctx = cvs.getContext('2d');
  ctx.clearRect(0, 0, cvs.width, cvs.height);
  const frame = S.renderViewIndices[idx];
  if (frame === undefined) return;
  const sx = S.tileW / S.W, sy = S.tileH / S.H;
  const fs = Math.max(10, Math.round(S.tileW / 56));
  const pad = Math.max(3, Math.round(S.tileW / 220));
  ctx.font = `bold ${fs}px system-ui`;
  ctx.textBaseline = 'top';
  // Each tile shows ONLY the boxes drawn on ITS OWN frame — plain 2D rects,
  // no cross-frame projection. Colour comes from the box's object.
  for (const b of boxesOnFrame(frame)){
    const col = colorOf(b); if (!col) continue;
    const minX = Math.max(0,          Math.floor(b.x1 * sx));
    const minY = Math.max(0,          Math.floor(b.y1 * sy));
    const maxX = Math.min(S.tileW - 1, Math.ceil(b.x2 * sx));
    const maxY = Math.min(S.tileH - 1, Math.ceil(b.y2 * sy));
    const w = maxX - minX, h = maxY - minY;
    if (w <= 0 || h <= 0) continue;
    // fill (matches canvas-fg's 0.12 alpha look)
    ctx.fillStyle = rgba(col, 0.12);
    ctx.fillRect(minX, minY, w, h);
    // outline
    ctx.strokeStyle = rgb(col);
    ctx.lineWidth = S.bboxWidth;
    ctx.strokeRect(minX + 0.5, minY + 0.5, w, h);
    // label tag
    const nm = objName(b);
    const tw = Math.ceil(ctx.measureText(nm).width);
    const lh = fs + 2 * pad;
    const ly = Math.max(minY - lh, 0);
    ctx.fillStyle = '#000';
    ctx.fillRect(minX, ly, tw + 2 * pad, lh);
    ctx.fillStyle = rgb(col);
    ctx.fillText(nm, minX + pad, ly + pad);
  }
}

function drawAllTileOverlays(){
  // Tile 0 is the input image (no overlay canvas); keyframe tiles start at 1.
  for (let i=1; i<nTiles(); i++) drawTileOverlay(i);
}

// Paint the per-tile border colour (matches the 3D frustum colour for that
// keyframe) and mark the active one. Called whenever activeKfIdx or
// keyframeColors change.
function applyTileStyles(){
  const t = document.getElementById('tiles');
  if (!t) return;
  for (let i=1; i<nTiles(); i++){   // tile 0 = input image: no border/active styling
    const div = t.children[i]; if (!div) continue;
    const col = S.keyframeColors[i];
    const active = (i === S.activeKfIdx);
    if (col){
      div.style.borderColor = `rgb(${col[0]},${col[1]},${col[2]})`;
    }
    div.style.borderWidth = active ? '4px' : '2px';
    // Bright white inner ring on the active one so it pops regardless of the
    // border colour landing in a dim part of the palette.
    div.style.boxShadow = active ? 'inset 0 0 0 2px rgba(255,255,255,0.85)' : 'none';
  }
}

async function selectKeyframe(i){
  if (i < 1) return;   // frame 0 (input image) is not selectable
  if (i === S.activeKfIdx) return;
  // Suppress the 800ms poll while we switch, so it can't revert activeFrame
  // before the server has recorded the new active keyframe.
  window._switchingKf = true;
  S.activeKfIdx = i;
  // Bind the left-bottom editor to this keyframe's frame: switch bg image and
  // show that frame's boxes.
  S.activeFrame = S.renderViewIndices[i];
  S.selectedBoxId = null;
  applyTileStyles();
  reloadBg();
  syncList(); draw();
  try {
    await fetch('/api/active_kf', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({idx: i}),
    });
  } catch (e) {}
  finally { window._switchingKf = false; }
}

function nTiles(){ return S.renderViewIndices.length || 9; }

function renderTiles(){
  const t = document.getElementById('tiles');
  const n = nTiles();
  if (t.children.length === 0){
    for (let i=0;i<n;i++){
      const div = document.createElement('div'); div.className='tile'; div.dataset.idx=i;
      const img = document.createElement('img'); img.id='tile-'+i;
      const cap = document.createElement('div'); cap.className='cap';
      if (i === 0){
        // Frame 0 = the input image: non-selectable, no overlay. Shown centered in
        // the top row at the same size as one keyframe tile (see .input-inner CSS).
        div.classList.add('input-tile');
        cap.textContent = 'input image';
        const inner = document.createElement('div'); inner.className='input-inner';
        inner.appendChild(img); inner.appendChild(cap);
        div.appendChild(inner);
      } else {
        div.onclick = () => selectKeyframe(i);
        const cvs = document.createElement('canvas'); cvs.id='tile-fg-'+i; cvs.className='tile-fg';
        div.appendChild(img); div.appendChild(cvs); div.appendChild(cap);
      }
      t.appendChild(div);
    }
  }
  // Only the BG <img> needs a fetch; overlays are drawn locally.
  const t0 = Date.now();
  document.getElementById('tile-0').src = '/img/input_image.jpg?t=' + t0;
  for (let i=1;i<n;i++)
    document.getElementById('tile-'+i).src = tileUrl(i, t0);
  applyTileStyles();
  drawAllTileOverlays();
}

// ── object list + per-frame box controls ─────────────────────────────────
// The right panel lists OBJECTS (identity across frames). Depth/thickness
// sliders act on the current object's box ON THE ACTIVE FRAME (they only
// affect the 3D lift for the viser scene; the 2D box is unchanged). Every
// input queues a debounced PATCH so slider tracking stays smooth.
function makeNumRow(box, key, min, max, label){
  const row = document.createElement('div'); row.className='ctrl';
  const lab = document.createElement('label'); lab.textContent = label;
  const rng = document.createElement('input'); rng.type='range';
  rng.min=min; rng.max=max; rng.step='any'; rng.value=box[key];
  const num = document.createElement('input'); num.type='number';
  num.min=min; num.max=max; num.step='0.01'; num.value=Number(box[key]).toFixed(2);
  rng.addEventListener('input', e => {
    const v = parseFloat(e.target.value);
    if (isNaN(v)) return;
    box[key] = v; num.value = v.toFixed(2);
    schedulePatch(box.box_id, {[key]: v});   // depth only moves the 3D lift
  });
  num.addEventListener('input', e => {
    const v = parseFloat(e.target.value);
    if (isNaN(v)) return;
    box[key] = v; rng.value = v;
    schedulePatch(box.box_id, {[key]: v});
  });
  row.appendChild(lab); row.appendChild(rng); row.appendChild(num);
  return row;
}

async function addObject(){
  const r = await fetch('/api/object', {method: 'POST'});
  const o = await r.json();
  S.objects.push(o);
  S.currentObjId = o.id;
  syncList();
  return o;
}

async function setCurrentObj(oid){
  // Suppress the poll while the select POSTs are in flight, else a poll can read
  // stale server state and revert the selection / miss the seeded box.
  window._selectingObj = true;
  try {
    S.currentObjId = oid;
    S.selectedBoxId = null;
    try { await fetch('/api/object/'+oid+'/current', {method:'POST'}); } catch (e) {}
    // If the object has no box on THIS frame but was placed on other frames, carry
    // it over from its nearest placed frame. The server reprojects the WORLD 3D
    // position into this frame's camera (so the object stays put in 3D instead of
    // just keeping the same 2D screen position).
    if (!S.boxes.some(b => b.obj_id === oid && b.frame === S.activeFrame)
        && S.boxes.some(b => b.obj_id === oid)){
      try {
        const r = await fetch('/api/box/carry', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({obj_id: oid, frame: S.activeFrame}),
        });
        const nb = await r.json();
        if (nb && nb.box_id !== undefined){
          const ex = boxById(nb.box_id);
          if (ex) Object.assign(ex, nb); else S.boxes.push(nb);
          S.selectedBoxId = nb.box_id;
        }
      } catch (e) {}
    }
    syncList();
    draw();   // editor now shows this object's box on the active frame
  } finally {
    window._selectingObj = false;
  }
}

async function renameObject(oid, name){
  const o = objById(oid); if (!o) return;
  o.name = (name || '').trim() || o.name;
  draw(); drawAllTileOverlays(); syncList();
  try {
    await fetch('/api/object/'+oid+'/rename', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({name: o.name}),
    });
  } catch (e) {}
}

async function deleteObject(oid){
  S.objects = S.objects.filter(o => o.id !== oid);
  S.boxes = S.boxes.filter(b => b.obj_id !== oid);
  if (S.currentObjId === oid) S.currentObjId = S.objects.length ? S.objects[0].id : null;
  S.selectedBoxId = null;
  syncList(); draw(); drawAllTileOverlays();
  try { await fetch('/api/object/'+oid, {method:'DELETE'}); } catch (e) {}
}

async function deleteBox(boxId){
  S.boxes = S.boxes.filter(b => b.box_id !== boxId);
  if (S.selectedBoxId === boxId) S.selectedBoxId = null;
  syncList(); draw(); drawAllTileOverlays();
  try { await fetch('/api/box/'+boxId, {method:'DELETE'}); } catch (e) {}
}

// ── explicit action buttons (Add / Rename / Delete object) ────────────────
async function actionAdd(){ await addObject(); }

function actionRename(){
  if (S.currentObjId === null){ alert('Add / select an object first.'); return; }
  // Trigger the in-place edit of the current object's name card (no popup).
  const el = document.querySelector('#bbox-list .name[data-oid="' + S.currentObjId + '"]');
  if (el && el._beginEdit) el._beginEdit();
}

async function actionDelete(){
  if (S.currentObjId === null){ alert('Add / select an object first.'); return; }
  await deleteObject(S.currentObjId);
}

function updateSelInfo(){
  const el = document.getElementById('sel-info'); if (!el) return;
  const o = objById(S.currentObjId);
  el.textContent = o ? `frame ${S.activeFrame} · current: ${o.name}`
                     : `frame ${S.activeFrame} · no object (＋ Add object)`;
}

function syncList(){
  updateSelInfo();
  // Never rebuild the list while a name is being edited in place: a direct
  // syncList() (e.g. setCurrentObj() after its awaited select POST resolves)
  // would blow away the contentEditable <span> mid-edit and drop focus, so the
  // rename silently fails. The poll already guards this via busy(); this guards
  // the direct callers too. On blur, renameObject() calls syncList() again with
  // _editingName cleared, so the list refreshes with the committed name.
  if (window._editingName) return;
  const list = document.getElementById('bbox-list'); list.innerHTML='';
  const empty = document.getElementById('bbox-popover-empty');
  if (empty) empty.style.display = S.objects.length === 0 ? '' : 'none';
  S.objects.forEach(o => {
    const boxHere = S.boxes.find(b => b.obj_id === o.id && b.frame === S.activeFrame);
    const nFrames = S.boxes.filter(b => b.obj_id === o.id).length;
    const card = document.createElement('div');
    card.className = 'bbox-card' + (o.id === S.currentObjId ? ' sel' : '');
    card.onclick = () => setCurrentObj(o.id);
    const head = document.createElement('div'); head.className='bbox-head';
    const sw = document.createElement('div'); sw.className='swatch';
    sw.style.background = rgb(o.color);
    // Click the card to select the object; double-click the name to rename it
    // IN PLACE (Enter / blur commits, Esc cancels) — no popup.
    const name = document.createElement('span'); name.className='name'; name.textContent=o.name;
    name.dataset.oid = o.id;
    name.style.flex='1'; name.style.minWidth='0'; name.style.cursor='pointer';
    name.style.padding='3px 4px'; name.style.borderRadius='4px';
    name.style.overflow='hidden'; name.style.textOverflow='ellipsis'; name.style.whiteSpace='nowrap';
    name.title = 'Double-click to rename';
    function beginEdit(){
      window._editingName = true;   // suppress the poll so syncList won't wipe the field
      name.contentEditable = 'true';
      name.style.cursor='text'; name.style.overflow='visible';
      name.style.textOverflow='clip';
      name.style.background='#0e0e12'; name.style.outline='1px solid #4079cf';
      name.focus();
      const sel = window.getSelection(), rng = document.createRange();
      rng.selectNodeContents(name); sel.removeAllRanges(); sel.addRange(rng);
    }
    name._beginEdit = beginEdit;
    name.ondblclick = e => { e.stopPropagation(); beginEdit(); };
    name.onclick = e => { if (name.contentEditable === 'true') e.stopPropagation(); };
    name.onkeydown = e => {
      if (name.contentEditable !== 'true') return;
      e.stopPropagation();   // don't leak typing to editor/FPS/modal key handlers
      if (e.key === 'Enter'){ e.preventDefault(); name.blur(); }
      else if (e.key === 'Escape'){ e.preventDefault(); name.textContent = o.name; name.blur(); }
    };
    name.onblur = () => {
      if (name.contentEditable !== 'true') return;
      name.contentEditable = 'false';
      window._editingName = false;
      name.style.cursor='pointer'; name.style.overflow='hidden';
      name.style.textOverflow='ellipsis';
      name.style.background='transparent'; name.style.outline='none';
      const nn = (name.textContent || '').trim();
      if (nn && nn !== o.name){ renameObject(o.id, nn); }
      else { name.textContent = o.name; }   // revert empty/unchanged
    };
    const del = document.createElement('button'); del.className='danger'; del.textContent='✕';
    del.title = 'Delete object (all frames)';
    del.onclick = (e) => { e.stopPropagation(); deleteObject(o.id); };
    head.appendChild(sw); head.appendChild(name); head.appendChild(del);
    card.appendChild(head);
    const meta = document.createElement('div'); meta.className='meta';
    meta.textContent = (boxHere ? 'on this frame' : 'not on this frame')
                       + ` · ${nFrames} frame(s) total`;
    card.appendChild(meta);
    if (boxHere){
      card.appendChild(makeNumRow(boxHere, 'front_depth', 0.1, 15, 'Front depth'));
      card.appendChild(makeNumRow(boxHere, 'thickness',   0.05, 5, 'Thickness'));
      const delb = document.createElement('button');
      delb.textContent = '✕ box on this frame'; delb.style.marginTop = '4px';
      delb.onclick = (e) => { e.stopPropagation(); deleteBox(boxHere.box_id); };
      card.appendChild(delb);
    }
    list.appendChild(card);
  });
}

// ── sync from server (skip while busy) ─────────────────────────────────
// Reconciles objects + boxes IN PLACE (keyed by id / box_id) so references
// captured in card input handlers stay valid across reloads.
async function loadState(){
  if (busy()) return;
  const r = await fetch('/api/state'); const s = await r.json();
  S.W = s.W; S.H = s.H; S.target_frame = s.target_frame;
  S.tileW = s.tile_w; S.tileH = s.tile_h;
  S.renderViewIndices = s.render_view_indices;
  S.keyframeColors = s.keyframe_colors || S.keyframeColors;
  if (s.pcd_point_size != null) S.pcdPointSize = s.pcd_point_size;
  if (window._genOnState) window._genOnState(s);
  if (window._upOnState) window._upOnState(s);
  const activeChanged = (s.active_kf_idx !== S.activeKfIdx);
  const curChanged = (s.current_obj_id !== S.currentObjId);
  S.activeKfIdx = s.active_kf_idx;
  S.activeFrame = s.active_frame;
  S.currentObjId = s.current_obj_id;
  const versionChanged = (s.render_version !== S.renderVersion);
  S.renderVersion = s.render_version;
  // objects
  const oMap = new Map(s.objects.map(o => [o.id, o]));
  const objStruct = S.objects.length !== s.objects.length
                    || S.objects.some(o => !oMap.has(o.id));
  S.objects = S.objects.filter(o => oMap.has(o.id));
  for (const o of S.objects){ Object.assign(o, oMap.get(o.id)); oMap.delete(o.id); }
  for (const no of oMap.values()) S.objects.push(no);
  // boxes
  const bMap = new Map(s.boxes.map(b => [b.box_id, b]));
  const boxStruct = S.boxes.length !== s.boxes.length
                    || S.boxes.some(b => !bMap.has(b.box_id));
  S.boxes = S.boxes.filter(b => bMap.has(b.box_id));
  for (const b of S.boxes){ Object.assign(b, bMap.get(b.box_id)); bMap.delete(b.box_id); }
  for (const nb of bMap.values()) S.boxes.push(nb);
  if (S.selectedBoxId !== null && !S.boxes.some(b => b.box_id === S.selectedBoxId))
    S.selectedBoxId = null;
  fitCanvas();
  if (objStruct || boxStruct) syncList(); else updateSelInfo();
  // Active keyframe changed elsewhere (e.g. a 3D frustum click on the viser
  // side) → rebind the editor to the new frame's image + boxes.
  if (activeChanged){ reloadBg(); syncList(); draw(); }
  else if (curChanged){ syncList(); draw(); }   // current object changed → refocus editor
  if (versionChanged){ renderTiles(); reloadBg(); }
  if (activeChanged || versionChanged) applyTileStyles();
  drawAllTileOverlays();
}

(function fpsTrajCapture(){
  if (typeof THREE === 'undefined') return;
  window._fpsBusy = false;
  const overlay = document.getElementById('fps-overlay');
  let canvas = document.getElementById('fps-canvas');
  const msgEl = document.getElementById('fps-msg');
  const capCount = document.getElementById('fps-cap-count');
  const enterBtn = document.getElementById('fps-enter');
  const btnExit = document.getElementById('fps-exit');
  const btnClr = document.getElementById('fps-clear-cap');
  const btnApply = document.getElementById('fps-apply-cap');
  if (!overlay || !canvas || !enterBtn) return;

  let renderer = null, scene = null, camera = null, ptsObj = null, markers = null;
  /** First keyframe / spawn pose from server — always shown (purple + white arrow). */
  let firstKfGroup = null;
  let caps = [];
  let yaw = 0, pitch = 0, raf = 0, alive = false, entering = false;
  let lastT = performance.now();
  const keys = new Set();
  const moveSp = 2.85, upSp = 1.85;
  const lookSp = 1.1;   // arrow-key look speed, rad/s
  const upWorld = new THREE.Vector3(0, -1, 0);
  const tmpF = new THREE.Vector3();
  const tmpR = new THREE.Vector3();
  const tmpP = new THREE.Vector3();
  /** Reused each look update — same pattern as Three.js ``PointerLockControls`` (read euler from quat, delta, ``setFromEuler`` on quat). */
  const eulFps = new THREE.Euler(0, 0, 0, 'YXZ');
  let lookDrag = false, lookLastX = 0, lookLastY = 0, lookPtrId = null;
  let touchLookId = null, touchLX = 0, touchLY = 0;
  let skipNextDragDelta = false;
  let skipLockMoveFrames = 0;

  function setMsg(t){ if (msgEl) msgEl.textContent = t || ''; }
  function syncCapUi(){ if (capCount) capCount.textContent = String(caps.length); }

  function disposeGroupDeep(grp){
    if (!grp) return;
    grp.traverse(ch => {
      if (ch.geometry){ ch.geometry.dispose(); }
      if (ch.material){
        if (Array.isArray(ch.material)) ch.material.forEach(m => m.dispose());
        else ch.material.dispose();
      }
    });
  }

  function setFirstKeyframeMarker(init){
    if (!scene || !init || !init.position) return;
    if (firstKfGroup){
      scene.remove(firstKfGroup);
      disposeGroupDeep(firstKfGroup);
      firstKfGroup = null;
    }
    const px = init.position[0], py = init.position[1], pz = init.position[2];
    firstKfGroup = new THREE.Group();
    const g = new THREE.SphereGeometry(0.11, 20, 20);
    const mat = new THREE.MeshBasicMaterial({color: 0xcc66ff, depthTest: true});
    const mesh = new THREE.Mesh(g, mat);
    mesh.position.set(px, py, pz);
    firstKfGroup.add(mesh);
    const dir = new THREE.Vector3(init.fwd[0], init.fwd[1], init.fwd[2]);
    if (dir.lengthSq() < 1e-10) dir.set(0, 0, 1);
    else dir.normalize();
    const origin = new THREE.Vector3(px, py, pz);
    const ah = new THREE.ArrowHelper(dir, origin, 0.48, 0xffffff, 0.14, 0.1);
    firstKfGroup.add(ah);
    scene.add(firstKfGroup);
  }

  function disposePoints(){
    if (!ptsObj || !scene) return;
    scene.remove(ptsObj);
    ptsObj.geometry.dispose();
    ptsObj.material.dispose();
    ptsObj = null;
  }

  function rebuildMarkers(){
    if (!scene) return;
    if (markers){
      scene.remove(markers);
      disposeGroupDeep(markers);
    }
    markers = new THREE.Group();
    caps.forEach((c, idx) => {
      const px = c.pos[0], py = c.pos[1], pz = c.pos[2];
      const g = new THREE.SphereGeometry(0.1, 18, 18);
      const mat = new THREE.MeshBasicMaterial({color: 0xffaa33, depthTest: true});
      const mesh = new THREE.Mesh(g, mat);
      mesh.position.set(px, py, pz);
      markers.add(mesh);
      const dir = new THREE.Vector3(c.fwd[0], c.fwd[1], c.fwd[2]);
      if (dir.lengthSq() < 1e-10) dir.set(0, 0, 1);
      else dir.normalize();
      const origin = new THREE.Vector3(px, py, pz);
      const ah = new THREE.ArrowHelper(dir, origin, 0.42, 0x44ffcc, 0.12, 0.09);
      markers.add(ah);
    });
    scene.add(markers);
    if (firstKfGroup) scene.add(firstKfGroup);
  }

  function onResize(){
    if (!renderer || !camera) return;
    const w = window.innerWidth, h = Math.max(1, window.innerHeight);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  function buildPoints(ab){
    const dv = new DataView(ab);
    const n = dv.getUint32(0, true);
    const pos = new Float32Array(ab, 4, n * 3);
    const cu8 = new Uint8Array(ab, 4 + n * 12, n * 3);
    const colF = new Float32Array(n * 3);
    for (let i = 0; i < n * 3; i++) colF[i] = cu8[i] / 255;
    const geom = new THREE.BufferGeometry();
    const posCopy = new Float32Array(pos);
    const colCopy = new Float32Array(colF);
    geom.setAttribute('position', new THREE.BufferAttribute(posCopy, 3));
    geom.setAttribute('color', new THREE.BufferAttribute(colCopy, 3));
    return new THREE.Points(geom, new THREE.PointsMaterial({
      // Match the "Point size" slider (state.pcd_point_size) so the first-person
      // capture view uses the same point size as the viser 3D / tiles.
      size: (S.pcdPointSize || 0.028),
      vertexColors: true, depthWrite: false, transparent: true, opacity: 0.9,
    }));
  }

  /** ``kind``: ``'drag'`` (px deltas), ``'lock'`` (movementXY), ``'wheel'`` (trackpad scroll),
   *  ``'keys'`` (arrow keys; deltas already in radians). */
  function rotateByDelta(dx, dy, kind){
    let mul;
    if (kind === 'wheel'){
      mul = 0.00038;
      dx = Math.max(-90, Math.min(90, dx));
      dy = Math.max(-90, Math.min(90, dy));
    } else if (kind === 'lock'){
      mul = 0.0022;
    } else if (kind === 'keys'){
      mul = 1;
    } else {
      mul = 0.0026;
    }
    const cap = kind === 'drag' ? 160 : kind === 'lock' ? 400 : 90;
    dx = Math.max(-cap, Math.min(cap, dx));
    dy = Math.max(-cap, Math.min(cap, dy));
    if (!camera) return;
    eulFps.setFromQuaternion(camera.quaternion, 'YXZ');
    eulFps.y -= dx * mul;
    eulFps.x -= dy * mul;
    const lim = 1.55;
    eulFps.x = Math.max(-lim, Math.min(lim, eulFps.x));
    eulFps.order = 'YXZ';
    camera.quaternion.setFromEuler(eulFps);
    yaw = eulFps.y;
    pitch = eulFps.x;
  }

  function freeze(){
    if (!camera) return;
    camera.getWorldDirection(tmpF);
    caps.push({
      pos: camera.position.toArray(),
      fwd: tmpF.toArray(),
    });
    syncCapUi();
    rebuildMarkers();
    setMsg('Pinned #' + caps.length + ' (need at least 2, then Apply trajectory)');
  }

  function tick(now){
    if (!alive || !camera || !renderer || !scene) return;
    const dt = Math.min(0.06, (now - lastT) / 1000);
    lastT = now;
    const sp = moveSp * dt;
    // WASD fly relative to the current view: W/S along the full view direction
    // (pitch included), A/D along the camera's own right axis. Space/Ctrl stay
    // on world up/down.
    camera.getWorldDirection(tmpF);
    tmpR.set(1, 0, 0).applyQuaternion(camera.quaternion);

    if (keys.has('KeyW')) camera.position.addScaledVector(tmpF, sp);
    if (keys.has('KeyS')) camera.position.addScaledVector(tmpF, -sp);
    if (keys.has('KeyA')) camera.position.addScaledVector(tmpR, -sp);
    if (keys.has('KeyD')) camera.position.addScaledVector(tmpR, sp);
    if (keys.has('Space')) camera.position.addScaledVector(upWorld, upSp * dt);
    if (keys.has('ControlLeft') || keys.has('ControlRight'))
      camera.position.addScaledVector(upWorld, -upSp * dt);
    // Arrow keys look around. rotateByDelta is drag-the-world (+dx turns the
    // view left, +dy tilts it up), hence the signs; pitch clamping is shared.
    const lookX = (keys.has('ArrowLeft') ? 1 : 0) - (keys.has('ArrowRight') ? 1 : 0);
    const lookY = (keys.has('ArrowUp') ? 1 : 0) - (keys.has('ArrowDown') ? 1 : 0);
    if (lookX || lookY) rotateByDelta(lookX * lookSp * dt, lookY * lookSp * dt, 'keys');

    renderer.render(scene, camera);
    raf = requestAnimationFrame(tick);
  }

  /** New `<canvas>` so the next `WebGLRenderer` gets a fresh context (reuse on same node is flaky). */
  function refreshFpsCanvasDom(){
    const par = canvas && canvas.parentNode;
    if (!par) return;
    const next = document.createElement('canvas');
    next.id = 'fps-canvas';
    par.replaceChild(next, canvas);
    canvas = next;
  }

  function bindFpsCanvasInteractions(){
    canvas.addEventListener('dblclick', () => { if (alive) canvas.requestPointerLock(); });
    canvas.addEventListener('pointerdown', e => {
      if (!alive || e.button !== 0) return;
      if (document.pointerLockElement === canvas) return;
      lookDrag = true;
      skipNextDragDelta = true;
      lookPtrId = e.pointerId;
      lookLastX = e.clientX;
      lookLastY = e.clientY;
      try { canvas.setPointerCapture(e.pointerId); } catch(_e){}
    });
    canvas.addEventListener('pointermove', e => {
      if (!alive || !lookDrag || document.pointerLockElement === canvas) return;
      if (skipNextDragDelta){
        skipNextDragDelta = false;
        lookLastX = e.clientX;
        lookLastY = e.clientY;
        return;
      }
      const dx = e.clientX - lookLastX, dy = e.clientY - lookLastY;
      lookLastX = e.clientX;
      lookLastY = e.clientY;
      if (dx !== 0 || dy !== 0) rotateByDelta(dx, dy, 'drag');
    });
    function endLookDrag(e){
      if (!lookDrag) return;
      if (lookPtrId !== null && e.pointerId !== lookPtrId) return;
      lookDrag = false;
      lookPtrId = null;
      try { canvas.releasePointerCapture(e.pointerId); } catch(_e){}
    }
    canvas.addEventListener('pointerup', endLookDrag);
    canvas.addEventListener('pointercancel', endLookDrag);
    canvas.addEventListener('wheel', e => {
      if (!alive) return;
      e.preventDefault();
      let dx = e.deltaX, dy = e.deltaY;
      if (e.deltaMode === 1){ dx *= 16; dy *= 16; }
      else if (e.deltaMode === 2){ dx *= 32; dy *= 32; }
      rotateByDelta(dx, dy, 'wheel');
    }, {passive: false});
    canvas.addEventListener('contextmenu', e => { if (alive) e.preventDefault(); });
    canvas.addEventListener('touchstart', e => {
      if (!alive || e.touches.length !== 1) return;
      touchLookId = e.touches[0].identifier;
      touchLX = e.touches[0].clientX;
      touchLY = e.touches[0].clientY;
    }, {passive: true});
    canvas.addEventListener('touchmove', e => {
      if (!alive || touchLookId === null) return;
      let t = null;
      for (let i = 0; i < e.touches.length; i++){
        if (e.touches[i].identifier === touchLookId){ t = e.touches[i]; break; }
      }
      if (!t) return;
      e.preventDefault();
      const dx = t.clientX - touchLX, dy = t.clientY - touchLY;
      touchLX = t.clientX;
      touchLY = t.clientY;
      rotateByDelta(dx, dy, 'drag');
    }, {passive: false});
    canvas.addEventListener('touchend', e => {
      if (touchLookId === null) return;
      let gone = false;
      for (let i = 0; i < e.changedTouches.length; i++){
        if (e.changedTouches[i].identifier === touchLookId){ gone = true; break; }
      }
      if (gone) touchLookId = null;
    });
  }

  async function enter(){
    if (alive || entering) return;
    entering = true;
    try{
    setMsg('');
    skipNextDragDelta = false;
    skipLockMoveFrames = 0;
    const init = await (await fetch('/api/fps_init')).json();
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a10);
    const fov = Math.max(20, Math.min(100, init.fov_y_deg || 55));
    camera = new THREE.PerspectiveCamera(fov, window.innerWidth / window.innerHeight, 0.04, 900);
    renderer = new THREE.WebGLRenderer({canvas, antialias: true, alpha: false});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    onResize();
    scene.add(new THREE.AmbientLight(0xffffff, 0.38));
    const dl = new THREE.DirectionalLight(0xffffff, 0.5);
    dl.position.set(2, 4, 3);
    scene.add(dl);

    if (init.has_pcd){
      try{
        const r = await fetch('/api/pcd_preview');
        if (r.ok){
          const ab = await r.arrayBuffer();
          ptsObj = buildPoints(ab);
          scene.add(ptsObj);
        } else setMsg('No point cloud (empty scene)');
      } catch(_e){ setMsg('Point cloud failed to load'); }
    } else setMsg('No point cloud (empty scene)');

    tmpP.set(init.position[0], init.position[1], init.position[2]);
    tmpF.set(init.fwd[0], init.fwd[1], init.fwd[2]).normalize();
    camera.position.copy(tmpP);
    camera.up.copy(upWorld);
    camera.lookAt(tmpP.x + tmpF.x, tmpP.y + tmpF.y, tmpP.z + tmpF.z);
    // Do not quaternion.setFromEuler here — with world up (0,-1,0) the YXZ
    // round-trip can flip the view. Only mirror yaw/pitch for bookkeeping.
    eulFps.setFromQuaternion(camera.quaternion, 'YXZ');
    yaw = eulFps.y;
    pitch = eulFps.x;

    caps = [];
    syncCapUi();
    rebuildMarkers();
    setFirstKeyframeMarker(init);

    overlay.classList.add('active');
    window._fpsBusy = true;
    alive = true;
    lastT = performance.now();
    window.addEventListener('resize', onResize);
    raf = requestAnimationFrame(tick);
    } catch (e) {
      setMsg(String(e));
      try { exit(); } catch (_e) {}
    } finally {
      entering = false;
    }
  }

  function exit(){
    alive = false;
    lookDrag = false;
    touchLookId = null;
    lookPtrId = null;
    skipNextDragDelta = false;
    skipLockMoveFrames = 0;
    cancelAnimationFrame(raf);
    raf = 0;
    overlay.classList.remove('active');
    window._fpsBusy = false;
    try { document.exitPointerLock(); } catch(_e){}
    window.removeEventListener('resize', onResize);
    disposePoints();
    if (firstKfGroup && scene){
      scene.remove(firstKfGroup);
      disposeGroupDeep(firstKfGroup);
      firstKfGroup = null;
    }
    if (markers && scene){
      scene.remove(markers);
      disposeGroupDeep(markers);
      markers = null;
    }
    if (renderer){
      renderer.dispose();
    }
    renderer = null; scene = null; camera = null;
    setMsg('');
    refreshFpsCanvasDom();
    bindFpsCanvasInteractions();
    fitLayout();
    window.dispatchEvent(new Event('resize'));
  }

  enterBtn.addEventListener('click', () => { enter().catch(e => setMsg(String(e))); });
  bindFpsCanvasInteractions();

  document.addEventListener('pointerlockchange', () => {
    if (!alive) return;
    if (document.pointerLockElement === canvas){
      lookDrag = false;
      skipLockMoveFrames = 2;
      setMsg('Pointer locked — move mouse to look (Esc to unlock)');
    } else {
      setMsg('Pointer unlocked — drag or scroll to look; double-click to lock again');
    }
  });

  document.addEventListener('mousemove', e => {
    if (!alive || document.pointerLockElement !== canvas) return;
    if (skipLockMoveFrames > 0){
      skipLockMoveFrames--;
      return;
    }
    rotateByDelta(e.movementX, e.movementY, 'lock');
  });

  document.addEventListener('keydown', e => {
    if (!alive) return;
    keys.add(e.code);
    if (e.code === 'Space' || e.code.startsWith('Arrow')) e.preventDefault();
    if (e.code === 'KeyF'){ e.preventDefault(); freeze(); }
    if (e.code === 'Escape'){ e.preventDefault(); exit(); }
  });
  document.addEventListener('keyup', e => { keys.delete(e.code); });

  btnExit.addEventListener('click', exit);
  btnClr.addEventListener('click', () => { caps = []; syncCapUi(); rebuildMarkers(); setMsg('Cleared pins'); });
  btnApply.addEventListener('click', async () => {
    if (caps.length < 2){ setMsg('Need at least 2 pinned poses'); return; }
    const orient = document.querySelector('input[name="fps-orient"]:checked').value === 'look_at' ? 'look_at' : 'tangent';
    setMsg('Submitting…');
    try{
      const r = await fetch('/api/traj_fps_apply', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({captures: caps, orient}),
      });
      const j = await r.json();
      setMsg(j.msg || '');
      if (j.ok){ await loadState(); exit(); }
    } catch(_e){ setMsg('Network error'); }
  });
})();

async function saveLayout(){
  const r = await fetch('/api/save_layout', {method:'POST'});
  const j = await r.json(); setStatus(j.msg || 'saved');
}

// ── brightness slider ────────────────────────────────────────────────────
// Brightness is baked into both the canvas-bg JPEG and the 8 tile JPEGs
// server-side via the `?b=` URL param — same code path, identical look.
// The slider stores the value and schedules a debounced refresh; rapid
// dragging gets coalesced into a single render burst.
(function(){
  const sl  = document.getElementById('bright');
  const out = document.getElementById('bright-val');
  let refreshTimer = null;
  function apply(){
    const v = parseFloat(sl.value);
    S.bright = v;
    out.textContent = v.toFixed(2);
    try { localStorage.setItem('viewer.bright', v.toFixed(2)); } catch(e) {}
    if (refreshTimer) clearTimeout(refreshTimer);
    refreshTimer = setTimeout(() => {
      refreshTimer = null;
      reloadBg(); renderTiles();
    }, 150);
  }
  const saved = (() => { try { return localStorage.getItem('viewer.bright'); } catch(e) { return null; } })();
  if (saved) sl.value = saved;
  sl.addEventListener('input', apply);
  apply();
})();

// ── bbox-width slider ────────────────────────────────────────────────────
// Outline width is purely client-side (canvas-fg + tile-fg overlays both read
// S.bboxWidth). No server roundtrip — instant feedback as you drag.
(function(){
  const sl  = document.getElementById('bbox-w');
  const out = document.getElementById('bbox-w-val');
  function apply(){
    const v = parseInt(sl.value, 10);
    if (isNaN(v)) return;
    S.bboxWidth = v;
    out.textContent = String(v);
    try { localStorage.setItem('viewer.bboxWidth', String(v)); } catch(e) {}
    draw();
    drawAllTileOverlays();
  }
  const saved = (() => { try { return localStorage.getItem('viewer.bboxWidth'); } catch(e) { return null; } })();
  if (saved) sl.value = saved;
  sl.addEventListener('input', apply);
  apply();
})();

// ── video generation (modal) ───────────────────────────────────────────────
// Opens a dialog to set prompt + seed, POSTs /api/generate (which persists the
// edited camera + layout and runs one in-process generation), polls
// /api/generate/status, then plays the result. Model-load status + the run
// button's enabled state are driven by _genOnState (called from loadState).
(function(){
  const ov = document.getElementById('gen-overlay');
  const openBtn = document.getElementById('gen-open');
  const closeBtn = document.getElementById('gen-close');
  const runBtn = document.getElementById('gen-run');
  const promptEl = document.getElementById('gen-prompt');
  const seedEl = document.getElementById('gen-seed');
  const stepsEl = document.getElementById('gen-steps');
  const cfgEl = document.getElementById('gen-cfg');
  const statusEl = document.getElementById('gen-status');
  const modelStatusEl = document.getElementById('gen-model-status');
  const progWrap = document.getElementById('gen-progress-wrap');
  const progBar = document.getElementById('gen-progress-bar');
  const videoEl = document.getElementById('gen-video');
  const resultRow = document.getElementById('gen-result-row');
  const dlEl = document.getElementById('gen-download');
  const seedUsedEl = document.getElementById('gen-seed-used');
  let caption = '';
  let promptTouched = false;
  let generating = false;
  let loadStatus = 'idle';
  let pollTimer = null;

  promptEl.addEventListener('input', () => { promptTouched = true; });

  function seedMode(){
    const el = document.querySelector('input[name="gen-seed-mode"]:checked');
    return el ? el.value : 'default';
  }
  function syncRunBtn(){ runBtn.disabled = generating || (loadStatus !== 'ready'); }

  // Called from loadState() on every /api/state poll.
  window._genOnState = function(s){
    if (s.caption !== undefined && s.caption !== null){
      caption = s.caption;
      if (!promptTouched && !promptEl.value) promptEl.value = caption;
    }
    if (s.default_seed !== undefined) document.getElementById('gen-seed-default').textContent = s.default_seed;
    loadStatus = s.gen_load_status || 'idle';
    let label = 'model: ' + loadStatus;
    if (loadStatus === 'loading') label = 'model: loading… ' + (s.gen_load_msg || '');
    else if (loadStatus === 'ready') label = 'model: ready ✓';
    else if (loadStatus === 'error') label = 'model: error — ' + (s.gen_load_msg || '');
    else if (loadStatus === 'unavailable') label = 'model: unavailable (no GPU backend)';
    modelStatusEl.textContent = label;
    syncRunBtn();
  };

  function openModal(){
    if (!promptTouched && !promptEl.value && caption) promptEl.value = caption;
    ov.classList.add('active');
  }
  function closeModal(){ ov.classList.remove('active'); }
  openBtn.addEventListener('click', openModal);
  closeBtn.addEventListener('click', closeModal);
  ov.addEventListener('click', (e) => { if (e.target === ov) closeModal(); });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && ov.classList.contains('active')) closeModal();
  });

  async function poll(jobId){
    let j;
    try {
      const r = await fetch('/api/generate/status?job=' + jobId);
      j = await r.json();
    } catch(e){ return; }  // transient; keep polling
    if (j.status === 'queued'){
      statusEl.textContent = 'queued…';
    } else if (j.status === 'generating'){
      const pct = Math.round((j.progress || 0) * 100);
      progWrap.style.display = 'block';
      progBar.style.width = pct + '%';
      statusEl.textContent = (j.msg || 'generating') + ' — ' + pct + '%';
    } else if (j.status === 'done'){
      progBar.style.width = '100%';
      statusEl.textContent = 'done ✓';
      videoEl.src = j.video + '&t=' + Date.now();
      videoEl.style.display = 'block';
      videoEl.load();
      videoEl.play().catch(() => {});
      dlEl.href = j.video;
      seedUsedEl.textContent = '';   // the seed is not shown with the result
      resultRow.style.display = 'flex';
      generating = false; syncRunBtn();
      clearInterval(pollTimer); pollTimer = null;
    } else if (j.status === 'error'){
      statusEl.textContent = 'error: ' + (j.msg || j.error || 'failed');
      progWrap.style.display = 'none';
      generating = false; syncRunBtn();
      clearInterval(pollTimer); pollTimer = null;
    }
  }

  runBtn.addEventListener('click', async () => {
    generating = true; syncRunBtn();
    statusEl.textContent = 'submitting…';
    progWrap.style.display = 'block';
    progBar.style.width = '0%';
    videoEl.style.display = 'none';
    resultRow.style.display = 'none';
    const payload = {
      seed_mode: seedMode(),
      seed: parseInt(seedEl.value, 10),
      prompt: promptEl.value,
      steps: parseInt(stepsEl.value, 10),
      cfg: parseFloat(cfgEl.value),
    };
    let j;
    try {
      const r = await fetch('/api/generate', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload),
      });
      j = await r.json();
    } catch(e){
      statusEl.textContent = 'request failed: ' + e;
      progWrap.style.display = 'none';
      generating = false; syncRunBtn();
      return;
    }
    if (!j.ok){
      statusEl.textContent = j.msg || 'failed to start';
      progWrap.style.display = 'none';
      generating = false; syncRunBtn();
      return;
    }
    statusEl.textContent = 'started…';
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(() => poll(j.job_id), 1200);
    poll(j.job_id);
  });
})();

// ── reset ──────────────────────────────────────────────────────────────────
// Clears the session server-side, then reloads the page so nothing client-side
// (result player, prompt, dialogs) survives. Two clicks, so a stray one is safe.
(function(){
  const btn = document.getElementById('reset-all');
  if (!btn) return;
  const label = btn.textContent;
  let armedUntil = 0;
  btn.addEventListener('click', async () => {
    if (Date.now() > armedUntil){
      armedUntil = Date.now() + 3000;
      btn.textContent = 'Reset? click again';
      setTimeout(() => { if (Date.now() >= armedUntil) btn.textContent = label; }, 3100);
      return;
    }
    armedUntil = 0;
    btn.disabled = true;
    btn.textContent = 'resetting…';
    try { await fetch('/api/reset', {method: 'POST'}); } catch(_e){}
    location.reload();
  });
})();

// ── image / folder upload → MoGe ────────────────────────────────────────────
(function(){
  const ov = document.getElementById('up-overlay');
  const openBtn = document.getElementById('up-open');
  const closeBtn = document.getElementById('up-close');
  const fileEl = document.getElementById('up-file');
  const preview = document.getElementById('up-preview');
  const promptEl = document.getElementById('up-prompt');
  const runBtn = document.getElementById('up-run');
  const statusEl = document.getElementById('up-status');
  let dataUrl = null, mogeStatus = 'idle';
  let autoOpened = false, uploading = false;
  let extra = {};          // optional folder payload: camera_npz (base64), layout (json text)
  const dirEl = document.getElementById('up-dir');
  const dirInfo = document.getElementById('up-dir-info');

  fileEl.addEventListener('change', () => {
    const f = fileEl.files && fileEl.files[0];
    if (!f) return;
    extra = {}; dirInfo.textContent = '';
    const rd = new FileReader();
    rd.onload = () => { dataUrl = rd.result; preview.src = dataUrl; preview.style.display = 'block'; };
    rd.readAsDataURL(f);
  });

  const readAs = (f, how) => new Promise((res, rej) => {
    const rd = new FileReader(); rd.onload = () => res(rd.result); rd.onerror = rej;
    if (how === 'text') rd.readAsText(f); else rd.readAsDataURL(f);
  });
  dirEl.addEventListener('change', async () => {
    const files = Array.from(dirEl.files || []);
    if (!files.length) return;
    const base = (f) => (f.name || '').toLowerCase();
    const pick = (pred) => files.find(pred);
    const img = pick(f => base(f) === 'first_frame.png') || pick(f => base(f) === 'input_image.png')
             || pick(f => /\.(png|jpe?g|webp)$/.test(base(f)) && !/pcd/.test(base(f)));
    const cam = pick(f => base(f) === 'camera.npz') || pick(f => base(f) === 'camera_da3.npz')
             || pick(f => /\.npz$/.test(base(f)) && !/pcd|edited/.test(base(f)));
    const lay = pick(f => base(f) === 'layout_lastframe.json') || pick(f => base(f) === 'layout_edited.json')
             || pick(f => /^layout.*\.json$/.test(base(f)) && !/track/.test(base(f)));
    const cap = pick(f => base(f) === 'caption.txt');
    extra = {};
    const found = [];
    if (img){ dataUrl = await readAs(img, 'url'); preview.src = dataUrl; preview.style.display = 'block'; found.push('image: ' + img.name); }
    if (cap){ promptEl.value = (await readAs(cap, 'text')).trim(); found.push('caption.txt'); }
    if (cam){ extra.camera_npz = (await readAs(cam, 'url')).split(',', 2)[1]; found.push('camera: ' + cam.name); }
    if (lay){ extra.layout = await readAs(lay, 'text'); found.push('layout: ' + lay.name); }
    dirInfo.textContent = found.length ? 'found ' + found.join(', ') : 'no usable files in that folder';
    if (!img) statusEl.textContent = 'folder has no image';
  });

  function openModal(){ ov.classList.add('active'); }
  function closeModal(){ ov.classList.remove('active'); }
  openBtn.addEventListener('click', openModal);
  closeBtn.addEventListener('click', closeModal);
  ov.addEventListener('click', (e) => { if (e.target === ov) closeModal(); });
  function syncRunBtn(){ runBtn.disabled = uploading || (mogeStatus !== 'ready'); }

  window._upOnState = function(s){
    mogeStatus = s.moge_load_status || 'idle';
    if (!uploading && mogeStatus !== 'ready')
      statusEl.textContent = 'MoGe ' + mogeStatus + (s.moge_load_msg ? ' — ' + s.moge_load_msg : '');
    if (!autoOpened && !s.has_scene){ autoOpened = true; openModal(); }
    syncRunBtn();
  };

  runBtn.addEventListener('click', async () => {
    if (!dataUrl){ statusEl.textContent = 'select an image first'; return; }
    uploading = true; syncRunBtn();
    statusEl.textContent = 'running MoGe…';
    let j;
    try {
      const r = await fetch('/api/upload_image', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(Object.assign({image: dataUrl, prompt: promptEl.value}, extra)),
      });
      j = await r.json();
    } catch(e){
      statusEl.textContent = 'request failed: ' + e;
      uploading = false; syncRunBtn(); return;
    }
    uploading = false; syncRunBtn();
    if (!j.ok){ statusEl.textContent = j.msg || 'upload failed'; return; }
    statusEl.textContent = 'done ✓ (' + j.n_points + ' pts, ' + j.resolution[0] + 'x' + j.resolution[1]
      + (j.n_poses ? ', ' + j.n_poses + ' camera poses' : '') + (j.n_boxes ? ', ' + j.n_boxes + ' boxes' : '') + ')';
    S.renderVersion = -1;         // force editor bg/tile refetch for the new scene
    await loadState();
    reloadBg(); renderTiles();
    setTimeout(closeModal, 800);
  });
})();

fitCanvas();
loadState();    // pulls camera matrices + bumps render_version → kicks off
                // BG fetch (renderTiles) and editor BG (reloadBg) on first hit.
// Single poller: /api/state is the source of truth. It only triggers a BG
// re-fetch when render_version bumps server-side (camera moved / brightness
// or splat radius changed). Bbox edits are pure-JS and never poll.
setInterval(() => { if (!busy()) loadState(); }, 800);
</script></body></html>
"""


# ────────────────────────────────────────────────────────────────────────────
# helpers: io, math, conventions
# ────────────────────────────────────────────────────────────────────────────

def placeholder_scene(Wp=640, Hp=360, Np=8):
    """The empty scene: a small grey still with identity cameras.
    Used at startup and whenever the session is reset."""
    frames = np.full((Np, Hp, Wp, 3), 28, np.uint8)
    intr = np.stack([np.array([[float(Wp), 0.0, Wp / 2.0],
                               [0.0, float(Wp), Hp / 2.0],
                               [0.0, 0.0, 1.0]])] * Np)
    extr = np.stack([np.eye(4)[:3, :4]] * Np)   # identity (w2c == c2w)
    return extr, intr, frames


def find_camera_file(clip_dir: Path):
    """camera_da3.npz (UI naming) or camera.npz (the LIFT examples/ naming)."""
    for name in ("camera_da3.npz", "camera.npz"):
        if (clip_dir / name).exists():
            return clip_dir / name
    raise FileNotFoundError(f"no camera_da3.npz / camera.npz in {clip_dir}")


def load_clip(clip_dir: Path):
    npz = np.load(find_camera_file(clip_dir))
    extr = np.asarray(npz["extrinsic"], dtype=np.float64)
    intr = np.asarray(npz["intrinsic"], dtype=np.float64)
    # Frames: prefer 0.mp4, else the first *.mp4 in the dir (some clips ship it as 1.mp4 etc.);
    # without any video, use the still image (first_frame.png / input_image.png) held for as
    # many frames as the trajectory has poses -- the same scene an upload of that folder gives.
    mp4 = clip_dir / "0.mp4"
    if not mp4.exists():
        cands = sorted(clip_dir.glob("*.mp4"))
        mp4 = cands[0] if cands else None
    if mp4 is not None:
        frames = iio.imread(mp4)
    else:
        img = next((clip_dir / n for n in ("first_frame.png", "input_image.png") if (clip_dir / n).exists()), None)
        if img is None:
            raise FileNotFoundError(f"no *.mp4 / first_frame.png / input_image.png in {clip_dir}")
        still = np.asarray(Image.open(img).convert("RGB"))
        frames = np.repeat(still[None], len(extr), axis=0)
    # Optional last-frame bbox annotation → pre-populates the editor's last-frame
    # boxes. Normalised to {"instances": [{"id", "category", "bbox":[x1,y1,x2,y2],
    # "caption"}, ...]} in video-pixel coords. Prefer layout_lastframe.json; else
    # derive it from a per-frame layout_track_*.json (its last frame — whose
    # `frames["<last>"]` list already has that exact per-instance shape).
    lastframe = None
    lf_path = clip_dir / "layout_lastframe.json"
    if lf_path.exists():
        lastframe = json.loads(lf_path.read_text())
    else:
        track_files = sorted(clip_dir.glob("layout_track*.json"))
        if track_files:
            fr = json.loads(track_files[0].read_text()).get("frames", {})
            if fr:
                key = str(len(frames) - 1)
                insts = fr.get(key) or fr.get(max(fr, key=lambda k: int(k)))
                if insts:
                    lastframe = {"instances": insts}
    caption_path = clip_dir / "caption.txt"
    caption = caption_path.read_text().strip() if caption_path.exists() else ""
    assert len(extr) == len(intr)
    return extr, intr, frames, lastframe, caption


def to_4x4(P34):
    M = np.eye(4)
    M[:3, :4] = P34
    return M


def quat_wxyz_from_R(Rmat):
    q = R.from_matrix(Rmat).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def R_from_quat_wxyz(wxyz):
    w, x, y, z = wxyz
    return R.from_quat([x, y, z, w]).as_matrix()


def compute_c2w(extr_3x4, convention):
    if convention == "c2w":
        return to_4x4(extr_3x4)
    return np.linalg.inv(to_4x4(extr_3x4))


def c2w_to_viser_pose(c2w_cv):
    """viser uses the OpenCV camera convention end-to-end (+Z forward, +X right,
    +Y down) — see add_camera_frustum() docstring. Pass c2w straight through;
    no convention flip. (Earlier this multiplied by diag(1,-1,-1,1) to go to
    OpenGL, which rotated every frustum 180° around X: apex pointed into the
    scene with the base behind the camera — the opposite of a real camera.)"""
    return quat_wxyz_from_R(c2w_cv[:3, :3]), c2w_cv[:3, 3]


def viser_pose_to_c2w(wxyz, position):
    c2w = np.eye(4)
    c2w[:3, :3] = R_from_quat_wxyz(wxyz)
    c2w[:3, 3] = np.asarray(position)
    return c2w


def fov_y_from_intrinsic(K, image_h):
    return 2.0 * np.arctan2(image_h / 2.0, float(K[1, 1]))


# ────────────────────────────────────────────────────────────────────────────
# point splat renderer
# ────────────────────────────────────────────────────────────────────────────

_DISK_OFFSETS_CACHE = {}

def _disk_offsets(radius):
    """All integer (dy, dx) offsets inside a filled disk of the given radius."""
    radius = max(0, int(radius))
    cached = _DISK_OFFSETS_CACHE.get(radius)
    if cached is not None:
        return cached
    offs = []
    r2 = radius * radius
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= r2:
                offs.append((dy, dx))
    arr = np.asarray(offs, dtype=np.int32)
    _DISK_OFFSETS_CACHE[radius] = arr
    return arr


def splat_render(pts_w, colors, c2w, K_full, full_hw, render_hw, radius=2):
    """Project points and paint a small filled disk at each one.

    ``radius=0`` paints a single pixel per point (the original behaviour);
    higher values make the cloud visibly dense even at modest point counts.
    Painting is done back-to-front so nearer points naturally occlude."""
    Hf, Wf = full_hw; Hr, Wr = render_hw
    sx, sy = Wr / Wf, Hr / Hf
    K = K_full.copy()
    K[0, 0] *= sx; K[1, 1] *= sy; K[0, 2] *= sx; K[1, 2] *= sy

    w2c = np.linalg.inv(c2w)
    cam = pts_w @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    valid = z > 1e-3
    cam = cam[valid]; col = colors[valid]; z = z[valid]
    if cam.shape[0] == 0:
        return np.zeros((Hr, Wr, 3), dtype=np.uint8)
    px = (cam[:, 0] / z) * K[0, 0] + K[0, 2]
    py = (cam[:, 1] / z) * K[1, 1] + K[1, 2]
    in_bound = (px >= 0) & (px < Wr) & (py >= 0) & (py < Hr)
    col = col[in_bound]; z = z[in_bound]
    px = px[in_bound].astype(np.int32); py = py[in_bound].astype(np.int32)
    order = np.argsort(-z)
    px = px[order]; py = py[order]; col = col[order]

    img = np.zeros((Hr, Wr, 3), dtype=np.uint8)
    if radius <= 0:
        img[py, px] = col
        return img
    for dy, dx in _disk_offsets(radius):
        pyy = py + dy
        pxx = px + dx
        m = (pxx >= 0) & (pxx < Wr) & (pyy >= 0) & (pyy < Hr)
        img[pyy[m], pxx[m]] = col[m]
    return img


_BRIGHT_LUTS = {}

def apply_brightness(img_u8, factor):
    """Gamma-style midtone lift on a uint8 RGB image (factor=1 is identity).

    Applied SERVER-side so any text/lines drawn on top stay saturated — a CSS
    `filter: brightness(...)` would wash those out too."""
    factor = float(factor)
    if factor <= 1.01:
        return img_u8
    factor = min(factor, 4.0)
    key = round(factor * 20)
    lut = _BRIGHT_LUTS.get(key)
    if lut is None:
        lut = np.clip(
            np.power(np.arange(256) / 255.0, 1.0 / factor) * 255.0,
            0, 255,
        ).astype(np.uint8)
        _BRIGHT_LUTS[key] = lut
    return cv2.LUT(img_u8, lut)


# ────────────────────────────────────────────────────────────────────────────
# 3D bbox candidates
# ────────────────────────────────────────────────────────────────────────────

PALETTE = [
    (230, 25, 75), (60, 180, 75), (40, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60),
    (170, 110, 40), (250, 190, 212),
]


def palette_color(i):
    return PALETTE[i % len(PALETTE)]


BBOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
              (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)]


def bbox_3d_corners_camera(x1, y1, x2, y2, front_depth, thickness, K):
    """Backproject 2D bbox into a 3D AABB in CAMERA coords.

    The 2D bbox + ``front_depth`` (depth of the *near face*) determines the front
    face. ``thickness`` extends the box back by that much along +Z.
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    zmin = float(front_depth)
    zmax = float(front_depth + thickness)
    # back-project at the front face
    p1 = np.array([(x1 - cx) / fx * zmin, (y1 - cy) / fy * zmin, zmin])
    p2 = np.array([(x2 - cx) / fx * zmin, (y2 - cy) / fy * zmin, zmin])
    xmin, xmax = sorted([float(p1[0]), float(p2[0])])
    ymin, ymax = sorted([float(p1[1]), float(p2[1])])
    return np.array([
        [xmin, ymin, zmin], [xmax, ymin, zmin],
        [xmax, ymax, zmin], [xmin, ymax, zmin],
        [xmin, ymin, zmax], [xmax, ymin, zmax],
        [xmax, ymax, zmax], [xmin, ymax, zmax],
    ], dtype=np.float64)


def project_corners_to_pixels(corners_w, K_full, c2w, full_hw, render_hw, near=0.05):
    """Project an 8-corner 3D bbox. Returns (8, 2) pixel coords or None.

    A bbox is skipped only if *every* corner lies behind the near plane;
    partially-behind boxes are still projected with a clamped depth so the
    visible part of the wireframe shows up (instead of disappearing entirely).
    """
    Hf, Wf = full_hw; Hr, Wr = render_hw
    sx, sy = Wr / Wf, Hr / Hf
    K = K_full.copy()
    K[0, 0] *= sx; K[1, 1] *= sy; K[0, 2] *= sx; K[1, 2] *= sy
    w2c = np.linalg.inv(c2w)
    cam = corners_w @ w2c[:3, :3].T + w2c[:3, 3]
    if (cam[:, 2] > near).sum() == 0:
        return None
    z = np.clip(cam[:, 2], near, None)
    px = (cam[:, 0] / z) * K[0, 0] + K[0, 2]
    py = (cam[:, 1] / z) * K[1, 1] + K[1, 2]
    return np.stack([px, py], axis=1)


def draw_3d_box_wireframe_2d(img, corners_2d, color, thickness=2, label=None):
    pts = corners_2d.astype(np.int32)
    for a, b in BBOX_EDGES:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, thickness, cv2.LINE_AA)
    if label:
        cv2.putText(img, label, (int(pts[0, 0]) + 2, max(int(pts[0, 1]) - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def render_condition_overlay(src_mp4, dst_mp4, layout, fps=16):
    """Write ``dst_mp4`` = ``src_mp4`` with each frame's 2D condition bbox +
    object name drawn on top, for the side-by-side "generated | + condition" view.

    The condition is exactly what fed generation (``layout_edited.json``): boxes
    live ONLY on the frames the user annotated, and the generated video has the
    same frame count, so frame index maps 1:1. Boxes are mapped from the layout's
    original resolution into the generated frame with the SAME max-scale +
    center-crop as ``batch_infer_camlayout.resize_bbox`` so they line up with the
    generated content. Colors are the objects' own UI colors (RGB), drawn on the
    RGB frames the same way the editor canvas does.
    """
    frames = iio.imread(src_mp4)                 # (T, H, W, 3) uint8 RGB
    T = len(frames)
    Ht, Wt = int(frames.shape[1]), int(frames.shape[2])
    ow, oh = layout.get("image_resolution", [Wt, Ht])
    ow, oh = float(ow), float(oh)
    scale = max(Wt / ow, Ht / oh)
    crop_x = (ow * scale - Wt) / 2.0
    crop_y = (oh * scale - Ht) / 2.0

    obj_meta = {o["id"]: (o.get("name") or f"obj{o['id']}",
                          tuple(int(c) for c in o.get("color", [255, 80, 80])))
                for o in layout.get("objects", [])}
    per_frame = {}
    for fs, boxes in layout.get("frames", {}).items():
        per_frame[int(fs)] = [(b["bbox_2d"], b["obj_id"]) for b in boxes]

    def _map_box(b):
        x1, y1, x2, y2 = (b[0] * scale - crop_x, b[1] * scale - crop_y,
                          b[2] * scale - crop_x, b[3] * scale - crop_y)
        x1, y1 = max(x1, 0.0), max(y1, 0.0)
        x2, y2 = min(x2, float(Wt)), min(y2, float(Ht))
        if x2 <= x1 or y2 <= y1:
            return None
        return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

    out = np.empty_like(frames)
    for t in range(T):
        img = np.ascontiguousarray(frames[t])    # cv2 draws in place
        for bbox, oid in per_frame.get(t, []):
            m = _map_box(bbox)
            if m is None:
                continue
            name, col = obj_meta.get(oid, (f"obj{oid}", (255, 80, 80)))
            x1, y1, x2, y2 = m
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
            cv2.putText(img, name, (x1 + 3, max(y1 - 5, 11)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        out[t] = img
    iio.imwrite(dst_mp4, out, fps=fps, codec="libx264")
    return str(dst_mp4)


def make_grid_4x2(tiles, captions=None, gap=4, bg=(30, 30, 30)):
    assert len(tiles) == 8
    h, w = tiles[0].shape[:2]
    out_h = 4 * h + 5 * gap
    out_w = 2 * w + 3 * gap
    grid = np.full((out_h, out_w, 3), bg, dtype=np.uint8)
    for idx, t in enumerate(tiles):
        r, c = divmod(idx, 2)
        y = gap + r * (h + gap); x = gap + c * (w + gap)
        grid[y:y + h, x:x + w] = t
        if captions is not None:
            cv2.putText(grid, captions[idx], (x + 4, y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return grid


def _resample_polyline_uniform(points, n_out):
    """Resample a polyline to ``n_out`` points uniformly by arc length.

    ``points`` is (M, d) with M ≥ 2."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return np.repeat(pts[:1], max(n_out, 1), axis=0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    sl = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(sl[-1])
    if total < 1e-12:
        return np.repeat(pts[:1], n_out, axis=0)
    targets = np.linspace(0.0, total, n_out)
    out = np.empty((n_out, pts.shape[1]), dtype=np.float64)
    for j, t in enumerate(targets):
        idx = int(np.searchsorted(sl, t, side="right") - 1)
        idx = max(0, min(idx, len(pts) - 2))
        denom = max(sl[idx + 1] - sl[idx], 1e-12)
        local = (t - sl[idx]) / denom
        out[j] = pts[idx] + local * (pts[idx + 1] - pts[idx])
    return out


def _rdp_simplify_2d(points, epsilon):
    """Ramer–Douglas–Peucker on an (M, 2) polyline."""
    pts = np.asarray(points, dtype=np.float64)
    m = len(pts)
    if m <= 2 or epsilon <= 0:
        return pts
    keep = np.zeros(m, dtype=bool)
    keep[0] = keep[-1] = True
    changed = True
    while changed:
        changed = False
        idxs = np.flatnonzero(keep)
        for ia, ib in zip(idxs[:-1], idxs[1:]):
            if ib <= ia + 1:
                continue
            a, b = pts[ia], pts[ib]
            ab = b - a
            L = float(np.linalg.norm(ab))
            if L < 1e-12:
                continue
            seg = pts[ia + 1:ib] - a
            dist = np.abs(seg[:, 0] * ab[1] - seg[:, 1] * ab[0]) / L
            mi = int(np.argmax(dist)) + ia + 1
            if float(dist.max()) > epsilon:
                keep[mi] = True
                changed = True
    return pts[keep]


def _chord_length_and_max_perp_dist_3d(pts):
    """Return (chord length a→b, max perpendicular distance of vertices)."""
    pts = np.asarray(pts, dtype=np.float64)
    a, b = pts[0], pts[-1]
    ab = b - a
    L = float(np.linalg.norm(ab))
    if L < 1e-12 or len(pts) < 2:
        return L, 0.0
    t = pts - a
    proj = (np.sum(t * ab, axis=1) / (L * L))[:, np.newaxis] * ab
    d = np.linalg.norm(t - proj, axis=1)
    return L, float(d.max())


def _smooth_resample_path_3d(pts_m3, n_out, smooth01):
    """Smooth a dense 3D stroke with a cubic B-spline (``splprep``) and
    resample to ``n_out`` points. ``smooth01`` in [0, 1] — higher = smoother.

    Endpoint "hooks" from raw ``splprep`` overshoot are avoided by (1) not
    letting moving-average padding move the first/last samples, (2) skipping
    the spline entirely for nearly straight polylines, and (3) pinning the
    fitted curve to match the polyline endpoints."""
    from scipy.interpolate import splprep, splev

    pts = np.asarray(pts_m3, dtype=np.float64).copy()
    if len(pts) < 2:
        return np.repeat(pts[:1], n_out, axis=0)
    smooth01 = float(np.clip(smooth01, 0.0, 1.0))
    end0, end1 = pts[0].copy(), pts[-1].copy()
    if len(pts) >= 5 and smooth01 > 0.04:
        kwin = max(1, int(0.12 * len(pts) * smooth01))
        ker = np.ones(2 * kwin + 1, dtype=np.float64)
        ker /= ker.sum()
        for dim in range(3):
            pts[:, dim] = np.convolve(pts[:, dim], ker, mode="same")
        pts[0], pts[-1] = end0, end1
    # Top-down strokes that are straight in XZ should not go through splprep —
    # even when Y varies along the stroke (lifted from the old path), the
    # smoothing spline can overshoot in XZ at the ends ("hooks").
    xz = pts[:, [0, 2]]
    a2, b2 = xz[0], xz[-1]
    ab2 = b2 - a2
    Lxz = float(np.linalg.norm(ab2))
    if Lxz > 1e-12:
        tv = xz - a2
        pv = (np.sum(tv * ab2, axis=1) / (Lxz * Lxz))[:, np.newaxis] * ab2
        devxz = float(np.linalg.norm(tv - pv, axis=1).max())
        if devxz < max(1e-5, 0.014 * Lxz):
            return _resample_polyline_uniform(pts, n_out)
    L, dev = _chord_length_and_max_perp_dist_3d(pts)
    if L < 1e-12:
        return np.repeat(pts[:1], n_out, axis=0)
    # Mouse "straight" strokes → skip spline (splprep + s>0 bends away from ends).
    if dev < max(1e-5, 0.014 * L):
        return _resample_polyline_uniform(pts, n_out)
    if len(pts) < 4:
        return _resample_polyline_uniform(pts, n_out)
    d = np.diff(pts, axis=0)
    var = max(float((d * d).sum()) / max(len(d), 1), 1e-12)
    s = (smooth01 ** 2) * var * len(pts) * 15.0
    try:
        kdeg = min(3, len(pts) - 1)
        tck, _u = splprep([pts[:, 0], pts[:, 1], pts[:, 2]], k=kdeg, s=s)
        uu = np.linspace(0.0, 1.0, n_out)
        out = np.stack(splev(uu, tck), axis=1).astype(np.float64)
        out[0] = pts[0]
        out[-1] = pts[-1]
        return out
    except Exception:
        return _resample_polyline_uniform(pts, n_out)


def _interp_capture_forwards(F, n_out):
    """Linearly interpolate K unit forward vectors to ``n_out`` samples."""
    F = np.asarray(F, dtype=np.float64)
    F = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-9)
    k = len(F)
    if k == 0:
        return np.zeros((n_out, 3), dtype=np.float64)
    if k == 1:
        return np.repeat(F[:1], n_out, axis=0)
    u = np.linspace(0.0, 1.0, n_out) * (k - 1)
    i0 = np.clip(np.floor(u).astype(np.int32), 0, k - 2)
    fr = (u - i0).astype(np.float64)[:, np.newaxis]
    out = F[i0] * (1.0 - fr) + F[i0 + 1] * fr
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
    return out


# ────────────────────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=Path, default=None,
                    help="clip directory to pre-load (video/image + camera .npz + caption.txt, optional "
                         "pcd_moge.npz / layout_lastframe.json). Its files are COPIED into the session "
                         "dir, so edits and generated videos never touch the source. Omit to start empty "
                         "and upload an image or a folder in the browser.")
    ap.add_argument("--session-root", type=Path, default=None,
                    help="where per-run session dirs go (uploads, edited camera/layout, generated videos); "
                         "default <UI>/.ui_sessions. Kept after exit unless --delete-session-on-exit.")
    ap.add_argument("--delete-session-on-exit", action="store_true")
    ap.add_argument("--default-seed", type=int, default=48,
                    help="seed used by the 'default' option of the generate panel (48 = the project-page seed of example1)")
    ap.add_argument("--port", default=8081, type=int,
                    help="single user-facing port — viser is reverse-proxied through this")
    ap.add_argument("--convention", choices=["c2w", "w2c"], default="w2c")
    ap.add_argument("--frustum-scale", default=0.15, type=float)
    ap.add_argument("--tile-width", default=720, type=int,
                    help="width (px) of each rendered tile in the 4x2 grid")
    ap.add_argument("--canvas-width", default=960, type=int,
                    help="width (px) of the last-frame bbox editor canvas")
    ap.add_argument("--pcd-max-points", default=120_000, type=int,
                    help="random-subsample the loaded point cloud to at most this many points")
    ap.add_argument("--splat-radius", default=2, type=int,
                    help="pixel radius for each splatted point in the 8 tiles + canvas "
                         "(0 = single pixel; larger fills the splat). Live-tweakable from "
                         "the viser GUI sidebar.")
    ap.add_argument("--initial-pullback", default=0.1, type=float,
                    help="initial camera distance behind frame 0, as a multiple of scene diagonal")
    args = ap.parse_args()

    # One session dir per run holds everything the editor reads and writes: the pre-loaded clip's
    # files (copied), uploads, MoGe outputs, the edited camera / layout, and generated videos.
    session_root = (args.session_root.resolve() if args.session_root
                    else (Path(__file__).resolve().parent.parent / ".ui_sessions"))
    session_root.mkdir(parents=True, exist_ok=True)
    clip_dir = session_root / f"session_{uuid.uuid4().hex[:10]}"
    clip_dir.mkdir(parents=True, exist_ok=True)
    preloaded = args.clip is not None
    lastframe_layout = None
    default_seed = int(args.default_seed)   # the "default" option of the generate panel
    if preloaded:
        src = args.clip.resolve()
        if not src.is_dir():
            ap.error(f"--clip: not a directory: {src}")
        for f in src.iterdir():
            if f.is_file():
                shutil.copy2(f, clip_dir / f.name)
        print(f"[viewer] pre-loading {src}  (working copy: {clip_dir})")
        extr, intr, frames, lastframe_layout, caption = load_clip(clip_dir)
        has_scene = True
    else:
        print(f"[viewer] empty scene — upload an image or a folder in the browser. session dir = {clip_dir}")
        extr, intr, frames = placeholder_scene()   # small grey still, identity cameras
        caption = ""
        has_scene = False
    N, H, W = frames.shape[:3]
    print(f"[viewer] N={N} frames, video {W}x{H}")

    # Pre-flip frames vertically (OpenCV +Y down -> viser frustum image plane +Y up).
    frames_flipped = np.stack([cv2.flip(f, 0) for f in frames], axis=0)

    # Thumbnails for frustum image overlay.
    thumb_w = 320
    thumb_h = int(round(H * thumb_w / W))
    thumbs_clean = np.stack([cv2.resize(frames_flipped[i], (thumb_w, thumb_h),
                                        interpolation=cv2.INTER_AREA) for i in range(N)])

    # Optional MoGe point cloud.
    pcd_path = clip_dir / "pcd_moge.npz"
    pcd = None
    pcd_diag = 0.0
    if pcd_path.exists():
        z = np.load(pcd_path, allow_pickle=True)
        pts = np.asarray(z["points"], dtype=np.float32)
        cols = np.asarray(z["colors"], dtype=np.uint8)
        if len(pts) > args.pcd_max_points:
            sel = np.random.default_rng(0).choice(len(pts), args.pcd_max_points,
                                                   replace=False)
            pts = pts[sel]; cols = cols[sel]
        pcd = {"points": pts, "colors": cols}
        pcd_diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        print(f"[viewer] loaded point cloud: {len(pts)} pts, scene diagonal ≈ {pcd_diag:.2f}")

    # Right-side panel = 9 frames: index 0 is frame 0 shown as the (non-selectable)
    # INPUT IMAGE at the top; indices 1..8 are the editable keyframes at frames
    # 9,19,29,39,49,59,69,80, laid out 4 rows x 2 cols. Frame 0 stays keyframes[0]
    # so the camera anchor/lock (enforce_first_keypoint_pose → keyframes[0]) is
    # unchanged. Each keyframe gets a palette color so tile border ↔ 3D frustum
    # pair up.
    render_view_indices = [min(f, N - 1)
                           for f in (0, 9, 19, 29, 39, 49, 59, 69, 80)]
    tile_w = args.tile_width
    tile_h = int(round(H * tile_w / W))
    canvas_w = args.canvas_width
    canvas_h = int(round(H * canvas_w / W))

    state = {
        "convention": args.convention,
        "c2w_orig": np.stack([compute_c2w(extr[i], args.convention) for i in range(N)]),
        "c2w_edit": None,
        "keyframes": list(render_view_indices),
        "active_kf_idx": 1,   # index 0 = frame 0 = input image (non-selectable)
        "frustum_scale": args.frustum_scale,
        "pcd_point_size": 0.020,
        "splat_radius": max(0, int(args.splat_radius)),
        "render_bg_cache": None,
        # Bumped whenever the cached BG renders (or the camera matrices they
        # depend on) go stale. The client polls this — when it changes, the
        # client refetches /img/tile_bg and rebuilds its camera matrix cache.
        # tile_bg_jpeg_cache: {(version, idx, bright_int): bytes}.
        "render_version": 0,
        "tile_bg_jpeg_cache": {},
        # Per-frame layout tracks. `objects` carry identity (id/name/colour)
        # across frames; `boxes` are the per-frame 2D rectangles, each tagged
        # with the object it belongs to and the video frame it was drawn on.
        # A box is only shown on its own frame (no cross-frame reprojection);
        # same obj_id across frames = the same object over time.
        "objects": [],         # [{id, name, color}]
        "boxes": [],           # [{box_id, obj_id, frame, x1,y1,x2,y2, front_depth, thickness}]
        "current_obj_id": None,
        "next_obj_id": 0,
        "next_box_id": 0,
        "target_frame": N - 1,  # last frame (splat radius / legacy paths)
    }
    state["c2w_edit"] = state["c2w_orig"].copy()

    # Debug: pre-populate the editor from the clip's last-frame bbox annotation
    # (layout_lastframe.json), if present. One object + one box per instance, on
    # the LAST frame; front_depth/thickness use the editor defaults. Selecting an
    # object later carries its box onto other frames to set per-frame motion.
    def seed_layout_from_lastframe(layout):
        """Pre-populate the editor from a last-frame layout ({"instances": [{bbox, caption|category}]},
        pixel coords of the CURRENT scene). One object + one box per instance on the last frame;
        front_depth/thickness use the editor defaults. Caller holds state_lock (or is single-threaded)."""
        lf = N - 1
        K_lf, c2w_lf = intr[lf], state["c2w_edit"][lf]
        DEF_FD, DEF_TH = 1.5, 1.0   # default depth / thickness
        for inst in (layout or {}).get("instances", []):
            bb = inst.get("bbox")
            if not bb or len(bb) != 4:
                continue
            x1 = min(max(float(bb[0]), 0.0), W); x2 = min(max(float(bb[2]), 0.0), W)
            y1 = min(max(float(bb[1]), 0.0), H); y2 = min(max(float(bb[3]), 0.0), H)
            if x2 <= x1 or y2 <= y1:
                continue
            oid = state["next_obj_id"]; state["next_obj_id"] += 1
            bid = state["next_box_id"]; state["next_box_id"] += 1
            name = str(inst.get("caption") or inst.get("category") or f"obj{oid}")
            cam = bbox_3d_corners_camera(x1, y1, x2, y2, DEF_FD, DEF_TH, K_lf)
            wc = (cam @ c2w_lf[:3, :3].T + c2w_lf[:3, 3]).tolist()
            state["objects"].append({"id": oid, "name": name,
                                     "color": list(palette_color(oid))})
            state["boxes"].append({
                "box_id": bid, "obj_id": oid, "frame": int(lf),
                "x1": int(round(x1)), "y1": int(round(y1)),
                "x2": int(round(x2)), "y2": int(round(y2)),
                "front_depth": DEF_FD, "thickness": DEF_TH, "world_corners": wc,
            })
        if state["objects"]:
            print(f"[viewer] pre-loaded {len(state['objects'])} last-frame "
                  f"bbox(es) from the layout annotation (frame {N - 1})")
        return len(state["objects"])

    if lastframe_layout:
        seed_layout_from_lastframe(lastframe_layout)

    def first_keypoint_frame_idx():
        """Video frame index for the first of the 8 UI keypoints (must stay at
        the loaded extrinsic so it matches frame 0 / the first tile)."""
        return int(state["keyframes"][0])

    def enforce_first_keypoint_pose():
        fi = first_keypoint_frame_idx()
        state["c2w_edit"][fi] = state["c2w_orig"][fi].copy()

    def camera_path_exists():
        """False while every camera is still the identity copy of frame 0 (a single uploaded
        image before any trajectory is applied or a keyframe edited)."""
        return not np.allclose(state["c2w_edit"], state["c2w_edit"][:1], atol=1e-6)

    # Bind viser to localhost only on an auto-picked free port; all external
    # traffic comes in through the wrapper's reverse proxy on args.port.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
        _s.bind(("127.0.0.1", 0))
        viser_port = _s.getsockname()[1]
    # Silence viser's startup banner (it prints internal HTTP/WS URLs via
    # rich.print, unconditionally). verbose=False only mutes connect logs.
    import rich as _rich
    _rich_print_orig = _rich.print
    _rich.print = lambda *a, **k: None
    try:
        server = viser.ViserServer(host="127.0.0.1", port=viser_port, verbose=False)
    finally:
        _rich.print = _rich_print_orig
    server.gui.configure_theme(control_width="large", dark_mode=True, show_logo=False)
    server.scene.set_up_direction("-y")
    server.scene.add_frame("/world", show_axes=True, axes_length=0.3, axes_radius=0.005)

    # Default 3D viewport: stand BEHIND frame 0, looking through the scene so the
    # user sees the camera trajectory + point cloud, not be immersed at frame 0.
    # viser is OpenCV everywhere: +Z forward, +Y down. So forward = c2w[:,2],
    # and world-up = -c2w[:,1] (negate camera-Y to get up).
    _c2w0 = state["c2w_edit"][0]
    _pos0 =  _c2w0[:3, 3]
    _fwd0 =  _c2w0[:3, 2]
    _up0  = -_c2w0[:3, 1]
    _pullback = max(args.initial_pullback * max(pcd_diag, 1.0), 2.0)
    _init_pos = _pos0 - _fwd0 * _pullback
    if pcd is not None:
        _look_at = pcd["points"].mean(axis=0)
    else:
        _look_at = _pos0 + _fwd0 * 2.0
    server.initial_camera.position = tuple(map(float, _init_pos))
    server.initial_camera.look_at  = tuple(map(float, _look_at))
    server.initial_camera.up       = tuple(map(float, _up0))
    server.initial_camera.fov      = float(fov_y_from_intrinsic(intr[0], H))

    @server.on_client_connect
    def _(client):
        # Apply the same initial pose to each newly-connected client (so a refresh
        # also resets the view, not just first-time visitors).
        client.camera.position = server.initial_camera.position
        client.camera.look_at  = server.initial_camera.look_at
        client.camera.up_direction = server.initial_camera.up
        client.camera.fov = server.initial_camera.fov

    handles = {
        "frustums": [],
        "trajectory": None,
        "active_gizmo": None,
        "active_big_frustum": None,
        "pcd": None,
        "bbox_3d": {},          # bbox_id -> line-segments handle
    }

    state_lock = threading.RLock()

    def invalidate_render():
        """Drop the splat-BG cache, drop the encoded-JPEG cache, bump the
        version so clients refetch on their next /api/state tick."""
        state["render_bg_cache"] = None
        state["tile_bg_jpeg_cache"] = {}
        state["render_version"] += 1

    # ── small helpers operating on state/handles ────────────────────────────
    def obj_by_id(oid):
        for o in state["objects"]:
            if o["id"] == oid:
                return o
        return None

    def lift_box_world(box):
        """Lift a per-frame 2D box into 8 world corners using ITS OWN frame's
        camera (not a shared target frame) so boxes on different frames sit at
        the right place in 3D. This is the 2D->3D placement math; its result is
        cached on box['world_corners'] at draw/edit time and THAT cache is the
        authoritative 3D (see box_world_corners), so carrying a box to another
        frame can keep the exact 3D box instead of re-deriving a drifted one."""
        fi = int(box["frame"])
        K = intr[fi]
        c2w = state["c2w_edit"][fi]
        cam = bbox_3d_corners_camera(box["x1"], box["y1"], box["x2"], box["y2"],
                                     box["front_depth"], box["thickness"], K)
        return cam @ c2w[:3, :3].T + c2w[:3, 3]

    def box_world_corners(box):
        """Authoritative 8 world corners of a box: the cached world_corners if
        present (so a box carried to another frame keeps the SOURCE frame's exact
        3D position), else a fresh lift for legacy boxes without the cache."""
        wc = box.get("world_corners")
        if wc is not None:
            return np.asarray(wc, dtype=np.float64)
        return lift_box_world(box)

    def build_layout_data():
        """Per-frame track export: objects (identity) + frames{frame: [boxes]}."""
        frames_out = {}
        for b in state["boxes"]:
            obj = obj_by_id(b["obj_id"])
            if obj is None:
                continue
            frames_out.setdefault(str(int(b["frame"])), []).append({
                "obj_id": b["obj_id"], "name": obj["name"],
                "bbox_2d": [b["x1"], b["y1"], b["x2"], b["y2"]],
                "front_depth": b["front_depth"], "thickness": b["thickness"],
                "world_corners": box_world_corners(b).tolist(),
            })
        return {
            "image_resolution": [int(W), int(H)],
            "objects": [{"id": o["id"], "name": o["name"], "color": list(o["color"])}
                        for o in state["objects"]],
            "frames": frames_out,
        }

    # ── scene rebuilds ──────────────────────────────────────────────────────
    def clear_frustums():
        for h in handles["frustums"]:
            try: h.remove()
            except Exception: pass
        handles["frustums"] = []

    def clear_handle(key):
        h = handles.get(key)
        if h is not None:
            try: h.remove()
            except Exception: pass
            handles[key] = None

    def clear_active():
        if handles["active_gizmo"] is not None:
            try: handles["active_gizmo"].remove()
            except Exception: pass
        handles["active_gizmo"] = None
        handles["active_big_frustum"] = None

    def rebuild_static():
        # Each keyframe gets a unique palette colour — the matching right-side
        # tile gets the same colour as its border, so the 3D frustum and the
        # 2D tile visibly pair up. Clicking the frustum makes that keyframe
        # active (same as clicking its tile or moving the slider).
        clear_frustums()
        active_idx = state["active_kf_idx"]
        for kf_idx, i in enumerate(state["keyframes"]):
            wxyz, pos = c2w_to_viser_pose(state["c2w_edit"][i])
            color = palette_color(kf_idx)
            is_active = (kf_idx == active_idx)
            h = server.scene.add_camera_frustum(
                f"/cams/f{i:03d}",
                fov=fov_y_from_intrinsic(intr[i], H),
                aspect=W / H,
                scale=state["frustum_scale"] * (1.4 if is_active else 1.0),
                wxyz=tuple(wxyz), position=tuple(pos),
                line_width=3.0 if is_active else 1.4,
                color=color,
            )

            # Bind kf_idx to the closure via a default arg so the handlers don't
            # end up pointing at the last loop value. Frame 0 (kf_idx 0, the input
            # image) is fixed and non-selectable, so it gets no click handler.
            if kf_idx >= 1:
                @h.on_click
                def _(_evt, _kf=kf_idx):
                    set_active_kf(_kf)

            handles["frustums"].append(h)

    def rebuild_trajectory():
        clear_handle("trajectory")
        centers = np.stack([state["c2w_edit"][i, :3, 3] for i in range(N)])
        if N >= 2:
            handles["trajectory"] = server.scene.add_spline_catmull_rom(
                "/traj", points=centers, line_width=3.0, color=(220, 60, 60)
            )

    def rebuild_active():
        # Only the transform_controls gizmo, no companion frustum — the static
        # blue frustums already mark the keyframe positions.
        clear_active()
        # First keypoint is locked to the loaded first-frame camera (matches video).
        if state["active_kf_idx"] == 0:
            return
        i = state["keyframes"][state["active_kf_idx"]]
        wxyz, pos = c2w_to_viser_pose(state["c2w_edit"][i])
        gizmo = server.scene.add_transform_controls(
            f"/edit/f{i:03d}",
            scale=max(0.2, state["frustum_scale"] * 2.5),
            wxyz=tuple(wxyz), position=tuple(pos), line_width=3.0,
        )
        handles["active_gizmo"] = gizmo
        handles["active_big_frustum"] = None

        @gizmo.on_update
        def _(_):
            new_c2w = viser_pose_to_c2w(np.asarray(gizmo.wxyz), np.asarray(gizmo.position))
            state["c2w_edit"][i] = new_c2w
            kf_pos = state["keyframes"].index(i)
            if kf_pos < len(handles["frustums"]):
                wxyz_s, pos_s = c2w_to_viser_pose(new_c2w)
                handles["frustums"][kf_pos].wxyz = tuple(wxyz_s)
                handles["frustums"][kf_pos].position = tuple(pos_s)
            rebuild_trajectory()
            # background point cloud renders depend on poses → invalidate.
            invalidate_render()
            rebuild_3d_bboxes()
            update_panels()

    def rebuild_pcd():
        clear_handle("pcd")
        if pcd is None:
            return
        handles["pcd"] = server.scene.add_point_cloud(
            "/pcd_moge",
            points=pcd["points"],
            colors=pcd["colors"],
            point_size=state["pcd_point_size"], point_shape="rounded",
        )

    def rebuild_3d_bboxes():
        for key, h in list(handles["bbox_3d"].items()):
            try: h.remove()
            except Exception: pass
            handles["bbox_3d"].pop(key, None)
        # Focus mode: show ONLY the current object's box on the ACTIVE frame.
        # So the 3D view declutters on frame switch (re-run via set_active_kf)
        # and reveals a box only once its object is selected (set_current_obj).
        obj = obj_by_id(state["current_obj_id"])
        if obj is None:
            return
        fi = active_frame_idx()
        for box in state["boxes"]:
            if box["obj_id"] != obj["id"] or int(box["frame"]) != fi:
                continue
            corners = box_world_corners(box)
            segs = np.array([[corners[a], corners[b]] for a, b in BBOX_EDGES])
            h = server.scene.add_line_segments(
                f"/bbox3d/{box['box_id']}", segs,
                colors=tuple(int(c) for c in obj["color"]),
                line_width=3.5,
            )
            handles["bbox_3d"][box["box_id"]] = h

    # ── render panels ───────────────────────────────────────────────────────
    def active_frame_idx():
        kfi = max(0, min(int(state["active_kf_idx"]), len(state["keyframes"]) - 1))
        return int(state["keyframes"][kfi])

    def render_canvas_now():
        fi = active_frame_idx()
        base = cv2.resize(frames[fi], (canvas_w, canvas_h),
                          interpolation=cv2.INTER_AREA)
        sx, sy = canvas_w / W, canvas_h / H
        for bb in state["boxes"]:
            if int(bb["frame"]) != fi:
                continue
            obj = obj_by_id(bb["obj_id"])
            if obj is None:
                continue
            x1, y1 = int(bb["x1"] * sx), int(bb["y1"] * sy)
            x2, y2 = int(bb["x2"] * sx), int(bb["y2"] * sy)
            cv2.rectangle(base, (x1, y1), (x2, y2), obj["color"], 2, cv2.LINE_AA)
            cv2.putText(base, obj["name"], (x1 + 3, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, obj["color"], 1, cv2.LINE_AA)
        return base

    def _ensure_bg_cache():
        """Populate state["render_bg_cache"] (one splat per render-view) if
        missing. Caller must hold state_lock."""
        if state["render_bg_cache"] is None:
            bgs = []
            if pcd is None:
                for vi in render_view_indices:
                    bgs.append(cv2.resize(frames[vi], (tile_w, tile_h),
                                          interpolation=cv2.INTER_AREA))
            else:
                radius = int(state.get("splat_radius", 2))
                for vi in render_view_indices:
                    bgs.append(splat_render(pcd["points"], pcd["colors"],
                                            state["c2w_edit"][vi], intr[vi],
                                            full_hw=(H, W),
                                            render_hw=(tile_h, tile_w),
                                            radius=radius))
            state["render_bg_cache"] = bgs

    def tile_bg_jpeg(idx, bright):
        """Encoded JPEG bytes of tile ``idx`` (splat BG + brightness, no bbox).
        Cached keyed by (render_version, idx, bright). The bbox overlay is
        drawn client-side on a <canvas> on top of this <img>."""
        bright_int = int(round(bright * 100))
        with state_lock:
            key = (state["render_version"], idx, bright_int)
            cached = state["tile_bg_jpeg_cache"].get(key)
            if cached is not None:
                return cached
            _ensure_bg_cache()
            bg = state["render_bg_cache"][idx]
        # Brightness + JPEG encode outside the lock (bg array is now stable
        # under the version key; future invalidate_render() would clear the
        # cache anyway, never mutating this snapshot).
        out = apply_brightness(bg, bright)
        if not out.flags.writeable:
            out = out.copy()
        data = jpeg_bytes(out, quality=88)
        with state_lock:
            # Recheck the key under lock — another thread may have rendered
            # the same tile concurrently; either result is identical.
            state["tile_bg_jpeg_cache"][key] = data
        return data

    def render_single_tile(idx, bright=1.0, bbox_width=3):
        """Tile idx (RGB uint8): point-cloud splat bg + bbox overlays.

        Bboxes are styled to match the client-side canvas overlay: a
        translucent interior fill on top of the splat, then a saturated
        outline at ``bbox_width`` px, then a black-tag label. The fill is a
        cheap numpy alpha blend so we don't pay for full RGBA→RGB composite
        per request; outlines/text go through Pillow for real TrueType."""
        with state_lock:
            radius = int(state.get("splat_radius", 2))
            if state["render_bg_cache"] is None:
                bgs = []
                if pcd is None:
                    for vi in render_view_indices:
                        bgs.append(cv2.resize(frames[vi], (tile_w, tile_h),
                                              interpolation=cv2.INTER_AREA))
                else:
                    for vi in render_view_indices:
                        bgs.append(splat_render(pcd["points"], pcd["colors"],
                                                state["c2w_edit"][vi], intr[vi],
                                                full_hw=(H, W),
                                                render_hw=(tile_h, tile_w),
                                                radius=radius))
                state["render_bg_cache"] = bgs
            bg = state["render_bg_cache"][idx]
            fi = render_view_indices[idx]
            # Only this frame's own boxes — no cross-frame reprojection.
            boxes_snapshot = [(dict(b), dict(obj_by_id(b["obj_id"]) or {}))
                              for b in state["boxes"] if int(b["frame"]) == fi]
            is_target = (fi == state["target_frame"])

        bbox_width = max(1, min(12, int(bbox_width)))
        pad        = max(3, int(round(tile_w / 220.0)))
        font_size  = max(12, int(round(tile_w / 52.0)))
        cap_size   = max(12, int(round(tile_w / 56.0)))
        FILL_ALPHA = 0.12          # matches canvas-fg's `rgba(c, 0.10)` look
        font       = get_font(font_size)
        cap_font   = get_font(cap_size)

        # Each box drawn at its own 2D coords (scaled to tile), no projection.
        sx, sy = tile_w / W, tile_h / H
        to_draw = []
        for bb, obj in boxes_snapshot:
            if not obj:
                continue
            color = tuple(int(c) for c in obj["color"])
            x1 = int(max(0,          np.floor(bb["x1"] * sx)))
            y1 = int(max(0,          np.floor(bb["y1"] * sy)))
            x2 = int(min(tile_w - 1, np.ceil( bb["x2"] * sx)))
            y2 = int(min(tile_h - 1, np.ceil( bb["y2"] * sy)))
            if x2 <= x1 or y2 <= y1:
                continue
            to_draw.append((x1, y1, x2, y2, color, obj["name"]))

        # Step 1: brightness + alpha-blended fills (numpy, in-place).
        tile = apply_brightness(bg, bright)
        if not tile.flags.writeable:
            tile = tile.copy()
        if to_draw:
            for x1, y1, x2, y2, color, _ in to_draw:
                region = tile[y1:y2, x1:x2].astype(np.float32)
                blended = region * (1.0 - FILL_ALPHA) + np.asarray(color, dtype=np.float32) * FILL_ALPHA
                tile[y1:y2, x1:x2] = blended.astype(np.uint8)

        # Step 2: outline + label via Pillow.
        im = Image.fromarray(tile)
        draw = ImageDraw.Draw(im)
        for x1, y1, x2, y2, color, name in to_draw:
            draw.rectangle([x1, y1, x2, y2], outline=color, width=bbox_width)
            tb = draw.textbbox((0, 0), name, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            box_h = th + 2 * pad
            ly = max(y1 - box_h, 0)
            draw.rectangle([x1, ly, x1 + tw + 2 * pad, ly + box_h],
                           fill=(0, 0, 0))
            draw.text((x1 + pad - tb[0], ly + pad - tb[1]), name,
                      fill=color, font=font)

        cap = f"key{idx + 1}" + ("  (target)" if is_target else "")
        tb = draw.textbbox((0, 0), cap, font=cap_font)
        cw, ch = tb[2] - tb[0], tb[3] - tb[1]
        draw.rectangle([0, 0, cw + 2 * pad, ch + 2 * pad], fill=(0, 0, 0))
        draw.text((pad - tb[0], pad - tb[1]), cap,
                  fill=(240, 240, 240), font=cap_font)

        return np.asarray(im)

    def update_panels():
        # All panels are now served via HTTP polling; nothing to push here.
        pass

    def full_rebuild():
        rebuild_static()
        rebuild_trajectory()
        rebuild_active()
        rebuild_pcd()
        invalidate_render()
        rebuild_3d_bboxes()
        update_panels()

    def apply_scene(new_extr, new_intr, new_frames, new_pcd, new_caption):
        """Swap the whole scene in place (used by image / folder upload).

        Because these are all main() locals shared as closure cells, rebinding
        them here (nonlocal) makes every other closure + Handler method see the
        new scene — no per-reference rewrite, and N/W/H may change. Resets the
        layout/trajectory state and rebuilds the viser scene."""
        nonlocal extr, intr, frames, caption, N, H, W
        nonlocal frames_flipped, thumb_h, thumbs_clean
        nonlocal pcd, pcd_diag, render_view_indices, tile_h, canvas_h, has_scene
        with state_lock:
            extr = np.asarray(new_extr, dtype=np.float64)
            intr = np.asarray(new_intr, dtype=np.float64)
            frames = np.asarray(new_frames)
            caption = new_caption or ""
            N, H, W = frames.shape[:3]
            frames_flipped = np.stack([cv2.flip(f, 0) for f in frames], axis=0)
            thumb_h = int(round(H * thumb_w / W))
            thumbs_clean = np.stack([cv2.resize(frames_flipped[i], (thumb_w, thumb_h),
                                                interpolation=cv2.INTER_AREA)
                                     for i in range(N)])
            pcd = new_pcd
            pcd_diag = (float(np.linalg.norm(pcd["points"].max(axis=0)
                                             - pcd["points"].min(axis=0)))
                        if pcd is not None and len(pcd["points"]) else 0.0)
            render_view_indices = [min(f, N - 1)
                                   for f in (0, 9, 19, 29, 39, 49, 59, 69, 80)]
            tile_h = int(round(H * tile_w / W))
            canvas_h = int(round(H * canvas_w / W))
            # Reset trajectory + layout state for the new scene.
            state["c2w_orig"] = np.stack([compute_c2w(extr[i], state["convention"])
                                          for i in range(N)])
            state["c2w_edit"] = state["c2w_orig"].copy()
            state["keyframes"] = list(render_view_indices)
            state["active_kf_idx"] = 1   # skip frame 0 (input image)
            state["objects"] = []
            state["boxes"] = []
            state["current_obj_id"] = None
            state["next_obj_id"] = 0
            state["next_box_id"] = 0
            state["target_frame"] = N - 1
            has_scene = True
        # Recenter the default 3D view on the new scene.
        c0 = state["c2w_edit"][0]
        pull = max(args.initial_pullback * max(pcd_diag, 1.0), 2.0)
        look = (pcd["points"].mean(axis=0) if pcd is not None and len(pcd["points"])
                else c0[:3, 3] + c0[:3, 2] * 2.0)
        server.initial_camera.position = tuple(map(float, c0[:3, 3] - c0[:3, 2] * pull))
        server.initial_camera.look_at = tuple(map(float, look))
        server.initial_camera.up = tuple(map(float, -c0[:3, 1]))
        server.initial_camera.fov = float(fov_y_from_intrinsic(intr[0], H))
        # Recenter already-connected clients too (on_client_connect only fires for new ones).
        try:
            for cl in server.get_clients().values():
                cl.camera.position = server.initial_camera.position
                cl.camera.look_at = server.initial_camera.look_at
                cl.camera.up_direction = server.initial_camera.up
                cl.camera.fov = server.initial_camera.fov
        except Exception:
            pass
        full_rebuild()

    def reset_session():
        """Clear everything and go back to the empty scene so a new image or folder can be uploaded
        (the layout, the edited trajectory and the previous result are dropped). To get the pre-loaded
        example back, upload its folder again or restart the server."""
        nonlocal has_scene
        pe, pi, pf = placeholder_scene()
        apply_scene(pe, pi, pf, None, "")
        with state_lock:
            has_scene = False
        print("[viewer] session reset", flush=True)

    # ── object / per-frame box CRUD (callable from HTTP handlers) ──────────
    def add_object():
        with state_lock:
            oid = state["next_obj_id"]; state["next_obj_id"] += 1
            obj = {"id": oid, "name": f"obj{oid}", "color": list(palette_color(oid))}
            state["objects"].append(obj)
            state["current_obj_id"] = oid
            rebuild_3d_bboxes()   # new object has no box yet → clears the 3D view
            return dict(obj)

    def set_current_obj(oid):
        with state_lock:
            if obj_by_id(oid) is None:
                return False
            state["current_obj_id"] = oid
            rebuild_3d_bboxes()   # refocus 3D on the newly-selected object
            return True

    def rename_object(oid, name):
        with state_lock:
            o = obj_by_id(oid)
            if o is None:
                return False
            o["name"] = str(name).strip() if str(name).strip() else f"obj{oid}"
            return True

    def remove_object(oid):
        with state_lock:
            state["objects"] = [o for o in state["objects"] if o["id"] != oid]
            gone = [b for b in state["boxes"] if b["obj_id"] == oid]
            state["boxes"] = [b for b in state["boxes"] if b["obj_id"] != oid]
            for b in gone:
                h = handles["bbox_3d"].pop(b["box_id"], None)
                if h is not None:
                    try: h.remove()
                    except Exception: pass
            if state["current_obj_id"] == oid:
                state["current_obj_id"] = (state["objects"][0]["id"]
                                           if state["objects"] else None)
            rebuild_3d_bboxes()   # current object may have changed → refocus
            return True

    def add_box(obj_id, frame, coords=None, world_corners=None):
        """Create (or move, if one already exists) the box for (obj_id, frame).
        One box per object per frame — drawing again just repositions it.

        ``world_corners`` (8x3), when given, is stored verbatim as the box's
        authoritative 3D — carry_box passes the SOURCE box's corners so the object
        keeps its exact 3D position across frames. Otherwise the 3D is derived from
        this frame's 2D placement (a fresh draw/placement on this frame)."""
        with state_lock:
            if obj_by_id(obj_id) is None:
                return None
            frame = int(frame)
            box = next((b for b in state["boxes"]
                        if b["obj_id"] == obj_id and int(b["frame"]) == frame), None)
            if box is None:
                bxid = state["next_box_id"]; state["next_box_id"] += 1
                box = {"box_id": bxid, "obj_id": obj_id, "frame": frame,
                       "x1": int(W * 0.4), "y1": int(H * 0.4),
                       "x2": int(W * 0.6), "y2": int(H * 0.6),
                       "front_depth": 1.5, "thickness": 1.0}
                state["boxes"].append(box)
            if coords:
                for k in ("x1", "y1", "x2", "y2"):
                    if k in coords:
                        box[k] = int(coords[k])
                for k in ("front_depth", "thickness"):   # seed depth when carrying over
                    if k in coords:
                        box[k] = float(coords[k])
            box["world_corners"] = ([[float(v) for v in c] for c in world_corners]
                                    if world_corners is not None
                                    else lift_box_world(box).tolist())
            rebuild_3d_bboxes()
            return dict(box)

    def carry_box(obj_id, target_frame):
        """Create the box for (obj_id, target_frame) by keeping the object's WORLD
        3D position from its nearest placed frame and reprojecting it into the
        target frame's camera. (Copying the 2D box directly would keep the screen
        position but drift the world position, since each frame's camera differs.)
        """
        with state_lock:
            if obj_by_id(obj_id) is None:
                return None
            tf = int(target_frame)
            existing = next((b for b in state["boxes"]
                             if b["obj_id"] == obj_id and int(b["frame"]) == tf), None)
            if existing is not None:
                return dict(existing)
            others = [b for b in state["boxes"] if b["obj_id"] == obj_id]
            if not others:
                return None
            others.sort(key=lambda b: abs(int(b["frame"]) - tf))
            src = others[0]
            corners_w = box_world_corners(src)          # 8 world corners of the placement
            K = intr[tf]
            c2w = state["c2w_edit"][tf]
            w2c = np.linalg.inv(c2w)
            cam = corners_w @ w2c[:3, :3].T + w2c[:3, 3]   # world -> target camera
            near = 0.05
            zc = cam[:, 2]
            zclip = np.clip(zc, near, None)
            u = (cam[:, 0] / zclip) * float(K[0, 0]) + float(K[0, 2])
            v = (cam[:, 1] / zclip) * float(K[1, 1]) + float(K[1, 2])
            x1 = min(max(float(np.min(u)), 0.0), W - 1)
            y1 = min(max(float(np.min(v)), 0.0), H - 1)
            x2 = min(max(float(np.max(u)), 1.0), W)
            y2 = min(max(float(np.max(v)), 1.0), H)
            if x2 <= x1:
                x2 = min(x1 + 1.0, W)
            if y2 <= y1:
                y2 = min(y1 + 1.0, H)
            front_depth = float(max(np.min(zc), near))
            thickness = float(max(np.max(zc) - np.min(zc), 1e-2))
            # The 2D rect + front_depth/thickness above are just the projection of
            # the source box into this frame (a display/edit handle + sensible
            # slider values). The object's 3D is kept EXACTLY by carrying the
            # source's world corners, so selecting it on another frame doesn't move
            # it in 3D. Dragging the 2D box (update_box) re-derives the 3D and sets
            # the per-frame motion.
            return add_box(obj_id, tf, {
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "front_depth": front_depth, "thickness": thickness,
            }, world_corners=corners_w)

    _ALLOWED_BOX_FIELDS = ("x1", "y1", "x2", "y2", "front_depth", "thickness")
    def update_box(box_id, fields):
        with state_lock:
            for b in state["boxes"]:
                if b["box_id"] == box_id:
                    for k, v in fields.items():
                        if k not in _ALLOWED_BOX_FIELDS:
                            continue
                        if k in ("x1", "y1", "x2", "y2"):
                            b[k] = int(v)
                        else:
                            b[k] = float(v)
                    # A user edit of the 2D box / depth is a deliberate re-placement
                    # on THIS frame → refresh the authoritative 3D from the new
                    # values (this is how per-frame motion is set).
                    b["world_corners"] = lift_box_world(b).tolist()
                    rebuild_3d_bboxes()
                    return True
            return False

    def remove_box(box_id):
        with state_lock:
            state["boxes"] = [b for b in state["boxes"] if b["box_id"] != box_id]
            h = handles["bbox_3d"].pop(box_id, None)
            if h is not None:
                try: h.remove()
                except Exception: pass

    # ── GUI layout ──────────────────────────────────────────────────────────
    def set_active_kf(idx):
        """Single entry point for changing the active keyframe — used by both
        3D frustum on_click and the POST /api/active_kf handler (tile click)."""
        # Clamp to [1, last]: index 0 is frame 0 (the input image), never selectable.
        idx = max(1, min(int(idx), len(state["keyframes"]) - 1))
        if idx == state["active_kf_idx"]:
            return
        state["active_kf_idx"] = idx
        rebuild_static(); rebuild_active()
        rebuild_3d_bboxes()   # active frame changed → refocus the 3D box

    with server.gui.add_folder("Display"):
        conv_dd = server.gui.add_dropdown(
            "Loaded extrinsic convention",
            options=("c2w", "w2c"),
            initial_value=state["convention"],
        )
        frustum_sz = server.gui.add_slider("Frustum scale", min=0.02, max=0.6, step=0.01,
                                            initial_value=state["frustum_scale"])

        @conv_dd.on_update
        def _(_):
            state["convention"] = conv_dd.value
            state["c2w_orig"] = np.stack([compute_c2w(extr[i], state["convention"]) for i in range(N)])
            state["c2w_edit"] = state["c2w_orig"].copy()
            full_rebuild()

        @frustum_sz.on_update
        def _(_):
            state["frustum_scale"] = float(frustum_sz.value)
            rebuild_static(); rebuild_active()

    with server.gui.add_folder("Point cloud (MoGe-2)"):
        # A pre-loaded clip without pcd_moge.npz is worth flagging; an empty scene simply has
        # no cloud until an image is uploaded, so don't warn.
        if pcd is None and preloaded:
            server.gui.add_markdown(
                "*No `pcd_moge.npz` in the clip: the point cloud is estimated with MoGe once it has loaded "
                "(or make one with `python viewer/estimate_pcd.py --clip <dir> --use-clip-intrinsic`).*"
            )
        # ALWAYS create the "Point size" control — it drives BOTH the viser 3D point
        # cloud (world-space radius) AND the tile splat (pixel radius). Mapping:
        # splat_px = round(point_size * 100), clamped to [0, 6] (default 0.020 m →
        # 2 px). With an empty scene the cloud doesn't exist yet (pcd is None at startup);
        # the slider must still be here so it works once MoGe swaps a cloud in — the
        # callback just no-ops (rebuild_pcd returns early) until then.
        pcd_size = server.gui.add_slider("Point size", min=0.002, max=0.05,
                                          step=0.001, initial_value=state["pcd_point_size"])

        @pcd_size.on_update
        def _(_):
            sz = float(pcd_size.value)
            state["pcd_point_size"] = sz
            rebuild_pcd()
            new_r = max(0, min(6, int(round(sz * 100))))
            if new_r != state["splat_radius"]:
                state["splat_radius"] = new_r
                invalidate_render()

    with server.gui.add_folder("Save"):
        save_traj = server.gui.add_button("Save camera trajectory → .npz")
        save_layout = server.gui.add_button("Save layout (bboxes) → .json")
        save_status = server.gui.add_markdown("*not saved yet*")

        @save_traj.on_click
        def _(_):
            out = _write_edited_camera()
            save_status.content = f"*camera →* `{out.name}`"
            print(f"[viewer] saved {out}")

        @save_layout.on_click
        def _(_):
            with state_lock:
                data = build_layout_data()
            out = clip_dir / "layout_edited.json"
            out.write_text(json.dumps(data, indent=2))
            save_status.content = (f"*layout →* `{out.name}` "
                                   f"({len(state['objects'])} objs, {len(state['boxes'])} boxes)")
            print(f"[viewer] saved {out}")

    # ── camera path: first-person (web) + spline helpers ───────────────────

    def _sample_spline(points, n):
        """Sample n positions along the curve through `points`. Uses scipy
        for ≥3 points (cubic if available, else quadratic); falls back to
        linear interpolation for 2 points."""
        pts = np.asarray(points, dtype=np.float64)
        if len(pts) < 2:
            return None
        if len(pts) == 2:
            ts = np.linspace(0.0, 1.0, n)
            return pts[0] + (pts[1] - pts[0]) * ts[:, None]
        try:
            from scipy.interpolate import splev, splprep
            k = min(3, len(pts) - 1)
            tck, _ = splprep([pts[:, 0], pts[:, 1], pts[:, 2]], k=k, s=0)
            u = np.linspace(0.0, 1.0, n)
            return np.stack(splev(u, tck), axis=1)
        except Exception:
            # Piecewise-linear fallback if scipy is missing or splprep barfs
            # on degenerate input (e.g. duplicate consecutive points).
            ts = np.linspace(0.0, len(pts) - 1, n)
            i0 = np.clip(np.floor(ts).astype(int), 0, len(pts) - 2)
            f = (ts - i0)[:, None]
            return pts[i0] + (pts[i0 + 1] - pts[i0]) * f

    def _rotation_from_forward(forward, world_up=np.array([0.0, -1.0, 0.0])):
        """Build a c2w rotation (right, down, forward columns) given a
        forward direction. OpenCV convention everywhere in this viewer:
        +X right, +Y down, +Z forward."""
        f = np.asarray(forward, dtype=np.float64)
        n = np.linalg.norm(f)
        if n < 1e-9:
            f = np.array([0.0, 0.0, 1.0])
        else:
            f = f / n
        # Pick a reference vector that isn't (nearly) parallel to forward,
        # so the cross product is well-defined.
        if abs(np.dot(f, world_up)) > 0.999:
            ref = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(f, ref)) > 0.999:
                ref = np.array([0.0, 0.0, 1.0])
            right = np.cross(ref, f)
        else:
            # world_down × forward = right (in OpenCV/RH coords)
            right = np.cross(-world_up, f)
        right /= max(np.linalg.norm(right), 1e-9)
        down = np.cross(f, right)
        down /= max(np.linalg.norm(down), 1e-9)
        return np.column_stack([right, down, f])

    def apply_samples_to_trajectory(samples, orient_mode, fwd_override=None):
        """Write ``N`` camera poses from polyline samples (positions + optional
        per-frame forward overrides for tangent mode)."""
        samples = np.asarray(samples, dtype=np.float64)
        if samples.shape[0] != N:
            t_old = np.linspace(0.0, 1.0, len(samples))
            t_new = np.linspace(0.0, 1.0, N)
            samples = np.stack(
                [np.interp(t_new, t_old, samples[:, j]) for j in range(3)], axis=1
            ).astype(np.float64)
        fwd_ov = None
        if fwd_override is not None:
            fwd_ov = np.asarray(fwd_override, dtype=np.float64)
            fwd_ov /= np.maximum(np.linalg.norm(fwd_ov, axis=1, keepdims=True), 1e-9)
        target = None
        if orient_mode == "look_at_scene":
            target = (pcd["points"].mean(axis=0) if pcd is not None
                      else state["c2w_edit"][:, :3, 3].mean(axis=0))
        new_c2w = np.tile(np.eye(4), (N, 1, 1))
        for i in range(N):
            pos = samples[i]
            if orient_mode == "look_at_scene":
                fwd = target - pos
            elif fwd_ov is not None:
                fwd = fwd_ov[i]
            else:
                if i == 0:
                    fwd = samples[1] - samples[0]
                elif i == N - 1:
                    fwd = samples[-1] - samples[-2]
                else:
                    fwd = samples[i + 1] - samples[i - 1]
            new_c2w[i, :3, :3] = _rotation_from_forward(fwd)
            new_c2w[i, :3, 3] = pos
        state["c2w_edit"] = new_c2w
        enforce_first_keypoint_pose()

    def apply_fps_trajectory_from_captures(captures, orient_mode):
        """``captures``: list of dicts with ``pos`` [x,y,z] and ``fwd`` [x,y,z]
        (world, OpenCV +Z forward). The locked first-frame camera is prepended
        as the trajectory's start (pin[0]) so frame 0 → the first user pin is
        interpolated instead of a hard cut. Interpolate to ``N`` cameras."""
        if len(captures) < 2:
            return "need at least 2 pinned poses (press F in first-person mode)"
        P = np.array([c["pos"] for c in captures], dtype=np.float64)
        F = np.array([c["fwd"] for c in captures], dtype=np.float64)
        F = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-9)
        # Seed the curve with the fixed first-frame camera so the path ramps out
        # of it. enforce_first_keypoint_pose() (called downstream) then snaps
        # frame 0's full rotation back to the exact loaded pose — the position
        # already matches this seed, so there's no jump. Skip when the first pin
        # already sits on the first-frame camera (a duplicate knot trips splprep).
        c2w0 = state["c2w_orig"][first_keypoint_frame_idx()]
        p0 = c2w0[:3, 3].astype(np.float64)
        f0 = c2w0[:3, 2].astype(np.float64)
        f0 = f0 / max(np.linalg.norm(f0), 1e-9)
        seeded = bool(np.linalg.norm(P[0] - p0) > 1e-3)
        if seeded:
            P = np.vstack([p0, P])
            F = np.vstack([f0, F])
        samples = _sample_spline(P, N)
        if samples is None:
            return "failed to interpolate positions from captures"
        fwd_ov = None
        if orient_mode == "tangent":
            fwd_ov = _interp_capture_forwards(F, N)
        apply_samples_to_trajectory(samples, orient_mode, fwd_override=fwd_ov)
        seed_note = " (+first-frame seed)" if seeded else ""
        return (f"set trajectory from {len(captures)} FP captures{seed_note} "
                f"→ {N} cameras")

    with server.gui.add_folder("Camera path"):
        server.gui.add_markdown(
            "**First-person (web):** use **First-person capture**, move with WASD, "
            "Space up / Ctrl down, look with the **arrow keys** or the mouse, "
            "press **F** to pin a pose (markers show in the 3D view). "
            "Add at least two pins, choose orientation, then **Apply trajectory**. "
            "Click any frustum to fine-tune that camera with the gizmo. "
            "The **first keyframe** (first frustum / leftmost tile) stays fixed to the "
            "loaded first-frame camera and cannot be moved."
        )
        restore_traj_btn = server.gui.add_button("↻ Restore original cameras")
        path_help = server.gui.add_markdown("*editing keyframe cameras*")

        @restore_traj_btn.on_click
        def _(_):
            state["c2w_edit"] = state["c2w_orig"].copy()
            full_rebuild()
            path_help.content = "*restored original camera trajectory*"

    full_rebuild()

    # ── HTTP wrapper server ─────────────────────────────────────────────────
    # JPEG params: 4:4:4 chroma (no subsampling) is what stops thin red/green
    # bbox lines from bleeding into the dark background and reading as pink.
    # Default 4:2:0 averages chroma across 2x2 px blocks → catastrophic for
    # saturated 1-2 px strokes.
    _jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    _sf = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_444", 0x111111)
    _sf_key = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR", None)
    if _sf_key is not None:
        _jpeg_params += [int(_sf_key), int(_sf)]

    def jpeg_bytes(rgb_img, quality=95):
        bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        params = list(_jpeg_params)
        params[1] = int(quality)
        ok, buf = cv2.imencode(".jpg", bgr, params)
        return buf.tobytes() if ok else b""

    def save_layout():
        with state_lock:
            data = build_layout_data()
        out = clip_dir / "layout_edited.json"
        out.write_text(json.dumps(data, indent=2))
        return (f"saved → {out.name} "
                f"({len(state['objects'])} objs, {len(state['boxes'])} boxes)")

    def _write_edited_camera():
        """Persist the edited trajectory to camera_da3_edited.npz (same schema /
        convention as the input camera_da3.npz). Shared by the viser Save button
        and the /api/generate endpoint."""
        edited = np.zeros_like(extr)
        for i in range(N):
            c2w = state["c2w_edit"][i]
            edited[i] = (c2w if state["convention"] == "c2w"
                         else np.linalg.inv(c2w))[:3, :4]
        out = clip_dir / "camera_da3_edited.npz"
        np.savez(out, extrinsic=edited.astype(np.float32),
                 intrinsic=intr.astype(np.float32))
        return out

    # ── video-generation backend (in-process; needs a GPU + the LIFT weights) ──
    # The editor runs fine without it (editing-only); generation is enabled when
    # UI/generate_single_video.py (a thin wrapper around the LIFT pipeline in
    # ../scripts/infer.py) is importable and a CUDA device is present. The Wan pipeline is loaded once
    # in a background thread so the editor is usable while it warms up.
    gen_lock = threading.Lock()          # serialises generations (single GPU)
    gen_jobs = {}                        # job_id -> {status, progress, msg, ...}
    gen_state = {"gen": None, "load_status": "idle", "load_msg": ""}

    def _find_gen_backend():
        """Directory holding generate_single_video.py: $GEN_BACKEND_DIR, else this UI dir."""
        here = Path(__file__).resolve()
        cands = []
        env = os.environ.get("GEN_BACKEND_DIR")
        if env:
            cands.append(Path(env))
        cands.append(here.parents[1])          # <repo>/UI/viewer/serve.py -> <repo>/UI
        for c in cands:
            if c and (c / "generate_single_video.py").is_file():
                return c
        return None

    # Some pipelines build the model inside `with torch.device('meta'):`, which sets
    # the PROCESS-GLOBAL default device to meta. MoGe builds its model with
    # `cls(**cfg)` then `load_state_dict(strict=False)`; if that construction runs
    # while the Wan load is inside the meta context, MoGe's params (esp. the DINOv2
    # backbone, absent from its checkpoint) land on meta and `.to(cuda)` fails with
    # "Cannot copy out of meta tensor". These two loads run in parallel background
    # threads, so serialise their model construction with a shared lock.
    model_init_lock = threading.Lock()

    def _load_model_bg():
        try:
            repo = _find_gen_backend()
            if repo is None:
                gen_state["load_status"] = "unavailable"
                gen_state["load_msg"] = ("generate_single_video.py not found "
                                         "(set GEN_BACKEND_DIR)")
                print(f"[gen] {gen_state['load_msg']}")
                return
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
            gen_state["load_status"] = "loading"
            gen_state["load_msg"] = "importing pipeline…"
            mod = importlib.import_module("generate_single_video")
            g = mod.VideoGenerator(ckpt_path=os.environ.get("WANGEN_CKPT"))

            def _log(m):
                gen_state["load_msg"] = str(m)
                print(m, flush=True)

            with model_init_lock:   # don't let MoGe construct during meta context
                g.load(log=_log)
            gen_state["gen"] = g
            gen_state["load_status"] = "ready"
            gen_state["load_msg"] = "ready"
            print("[gen] model ready")
        except Exception as e:
            gen_state["load_status"] = "error"
            gen_state["load_msg"] = f"load failed: {e}"
            traceback.print_exc()

    if os.environ.get("VIEWER_ENABLE_GEN", "1") != "0":
        threading.Thread(target=_load_model_bg, daemon=True).start()

    # ── MoGe backend (uploads: image -> point cloud + first-frame camera) ──
    moge_state = {"runner": None, "load_status": "idle", "load_msg": ""}
    moge_lock = threading.Lock()   # serialise uploads (single GPU)

    def _load_moge_bg():
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            moge_state["load_status"] = "loading"
            moge_state["load_msg"] = "importing MoGe…"
            from moge_runner import MoGeRunner
            r = MoGeRunner(device="cuda")

            def _log(m):
                moge_state["load_msg"] = str(m)
                print(m, flush=True)

            with model_init_lock:   # serialise vs the video-model load
                r.load(log=_log)
            moge_state["runner"] = r
            moge_state["load_status"] = "ready"
            moge_state["load_msg"] = "ready"
            _estimate_preloaded_pcd(r)
        except Exception as e:
            moge_state["load_status"] = "error"
            moge_state["load_msg"] = f"MoGe load failed: {e}"
            traceback.print_exc()

    def _estimate_preloaded_pcd(runner):
        """A pre-loaded clip without pcd_moge.npz: lift its first frame with MoGe (same recipe as
        estimate_pcd.py --use-clip-intrinsic), swap the cloud into the live scene and save it to the
        session dir so the file-based paths see it too."""
        nonlocal pcd, pcd_diag
        if not preloaded or pcd is not None:
            return
        try:
            with state_lock:
                img = np.ascontiguousarray(frames[0])
                K0, ex0 = intr[0].copy(), extr[0].copy()
                conv = state["convention"]
            fov_x = float(np.degrees(2.0 * np.arctan2(img.shape[1] / 2.0, K0[0, 0])))
            print(f"[moge] no pcd_moge.npz in the pre-loaded clip -> estimating (fov_x hint {fov_x:.1f}°)", flush=True)
            with moge_lock:
                pts, cols, _ = runner.infer_image(img, max_points=args.pcd_max_points, fov_x_deg=fov_x)
            c2w0 = compute_c2w(ex0, conv)
            pts = (pts @ c2w0[:3, :3].T + c2w0[:3, 3]).astype(np.float32)
            np.savez(clip_dir / "pcd_moge.npz", points=pts, colors=cols.astype(np.uint8))
            with state_lock:
                pcd = {"points": pts, "colors": cols}
                pcd_diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
            rebuild_pcd()
            invalidate_render()
            print(f"[moge] point cloud ready: {len(pts)} pts, scene diagonal ≈ {pcd_diag:.2f}", flush=True)
        except Exception as e:
            print(f"[moge] auto point-cloud estimation failed: {e}", flush=True)
            traceback.print_exc()

    if os.environ.get("VIEWER_ENABLE_GEN", "1") != "0":
        threading.Thread(target=_load_moge_bg, daemon=True).start()

    def _handle_upload(img_rgb, prompt="", camera=None, layout=None):
        """img_rgb: (H,W,3) uint8. Run MoGe → build an 81-frame scene → persist the
        tmp clip (0.mp4/camera_da3.npz/pcd_moge.npz) → swap the live scene in.

        Optional folder payload (the LIFT examples/ format):
          camera: {"extrinsic": (N,3|4,4) w2c, "intrinsic": (N,3,3) px at the image's
                  resolution} → used as the trajectory instead of the identity camera
                  (the MoGe cloud is lifted from frame 0 into that trajectory's world);
          layout: {"instances": [{"bbox": [x1,y1,x2,y2] px, "caption"|"category"}]}
                  → pre-populates the last-frame boxes.
        Returns a status dict."""
        runner = moge_state["runner"]
        if runner is None:
            raise RuntimeError(moge_state.get("load_msg") or "MoGe not loaded")
        # Cap the working resolution (keep aspect, even dims) to bound compute.
        H0, W0 = img_rgb.shape[:2]
        max_side = 1280
        scale = min(1.0, max_side / float(max(H0, W0)))
        Wt = int(round(W0 * scale)) // 2 * 2
        Ht = int(round(H0 * scale)) // 2 * 2
        sx, sy = Wt / float(W0), Ht / float(H0)
        if (Wt, Ht) != (W0, H0):
            img_rgb = cv2.resize(img_rgb, (Wt, Ht), interpolation=cv2.INTER_AREA)
        fov_hint = None
        if camera is not None:
            K0 = np.asarray(camera["intrinsic"], dtype=np.float64)[0]
            fov_hint = float(np.degrees(2.0 * np.arctan2(W0 / 2.0, K0[0, 0])))
        pts, cols, K = runner.infer_image(img_rgb, max_points=args.pcd_max_points, fov_x_deg=fov_hint)
        if camera is None:
            n_frames = 81
            new_intr = np.stack([K] * n_frames)                       # (81,3,3)
            new_extr = np.stack([np.eye(4)[:3, :4]] * n_frames)       # identity w2c
        else:
            ex = np.asarray(camera["extrinsic"], dtype=np.float64)[:, :3, :4]
            it = np.asarray(camera["intrinsic"], dtype=np.float64).copy()
            it[:, 0, :] *= sx; it[:, 1, :] *= sy                      # intrinsics follow the resize
            n_frames = len(ex)
            new_extr, new_intr = ex, it
            # MoGe points are in frame-0 camera coords; move them into the trajectory's world.
            c2w0 = compute_c2w(ex[0], state["convention"])
            pts = (pts @ c2w0[:3, :3].T + c2w0[:3, 3]).astype(np.float32)
        new_frames = np.repeat(img_rgb[None], n_frames, axis=0)   # (N,H,W,3)
        new_pcd = {"points": pts, "colors": cols}
        # Persist to the session dir so the file-based generation path works. The still image is
        # saved LOSSLESSLY as input_image.png (the generator conditions on that; the H.264 0.mp4 below
        # would alter it by a few grey levels and change the result for a given seed) and, for the
        # viewer, as 0.mp4 = the image held for N frames.
        Image.fromarray(img_rgb).save(clip_dir / "input_image.png")
        iio.imwrite(clip_dir / "0.mp4", new_frames, fps=16, codec="libx264")
        np.savez(clip_dir / "camera_da3.npz",
                 extrinsic=new_extr.astype(np.float32),
                 intrinsic=new_intr.astype(np.float32))
        np.savez(clip_dir / "pcd_moge.npz",
                 points=pts.astype(np.float32), colors=cols.astype(np.uint8))
        (clip_dir / "caption.txt").write_text(prompt or "")
        apply_scene(new_extr, new_intr, new_frames, new_pcd, prompt or "")
        n_boxes = 0
        if layout:
            insts = []
            for inst in layout.get("instances", []):
                bb = inst.get("bbox")
                if bb and len(bb) == 4:
                    insts.append(dict(inst, bbox=[bb[0] * sx, bb[1] * sy, bb[2] * sx, bb[3] * sy]))
            with state_lock:
                n_boxes = seed_layout_from_lastframe({"instances": insts})
            rebuild_3d_bboxes()
            invalidate_render()
        return {"ok": True, "n_points": int(len(pts)), "resolution": [Wt, Ht],
                "n_poses": int(n_frames), "n_boxes": int(n_boxes)}

    def _cleanup_session():
        shutil.rmtree(clip_dir, ignore_errors=True)
        print(f"[viewer] deleted session dir {clip_dir}")

    if args.delete_session_on_exit:
        atexit.register(_cleanup_session)
        # Ensure atexit runs on SIGTERM too (Ctrl-C already raises KeyboardInterrupt).
        try:
            signal.signal(signal.SIGTERM, lambda *_a: (_ for _ in ()).throw(SystemExit(0)))
        except Exception:
            pass

    def _run_generation(job_id, params):
        job = gen_jobs[job_id]
        try:
            g = gen_state["gen"]
            if g is None:
                raise RuntimeError(gen_state.get("load_msg") or "model not loaded")
            # Persist the current camera + layout to the clip dir, then generate.
            with state_lock:
                _write_edited_camera()
                data = build_layout_data()
            (clip_dir / "layout_edited.json").write_text(json.dumps(data, indent=2))
            out_path = clip_dir / "generated" / f"{job_id}.mp4"

            def _pcb(frac, msg):
                job["progress"] = float(frac)
                job["msg"] = msg

            job["status"] = "generating"
            job["msg"] = "preprocessing…"
            g.generate(
                clip_dir=clip_dir,
                out_path=str(out_path),
                seed=params["seed"],
                prompt=params.get("prompt") or None,
                num_frames=params.get("num_frames", 81),
                height=params.get("height", 352),
                width=params.get("width", 640),
                fps=params.get("fps", 16),
                num_inference_steps=params.get("steps", 50),
                cfg_scale=params.get("cfg", 6.0),
                progress_cb=_pcb,
            )
            job["video"] = str(out_path)
            job["status"] = "done"
            job["progress"] = 1.0
            job["msg"] = "done"
        except Exception as e:
            job["status"] = "error"
            job["msg"] = f"error: {e}"
            job["error"] = str(e)
            traceback.print_exc()
        finally:
            try:
                gen_lock.release()
            except RuntimeError:
                pass

    wrapper_html_filled = WRAPPER_HTML

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a, **_kw): pass  # silence default per-request log

        def _send(self, code, body=b"", ctype="text/plain", extra_headers=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if extra_headers:
                for k, v in extra_headers.items():
                    self.send_header(k, v)
            self.end_headers()
            if body: self.wfile.write(body)

        # ── reverse proxy to the internal viser server ──────────────────────
        def _viser_target_path(self):
            """Rewrite incoming path before forwarding to viser.

            The wrapper page lives at `/`, so the iframe loads viser at
            `/__viser/` — that prefix is stripped here. Everything else (asset
            paths emitted by viser's own HTML, WebSocket paths) passes through
            unchanged."""
            p = self.path
            q = p.find("?")
            base, qs = (p[:q], p[q:]) if q >= 0 else (p, "")
            if base == "/__viser" or base == "/__viser/":
                return "/" + qs
            if base.startswith("/__viser/"):
                return base[len("/__viser"):] + qs
            return p

        def _proxy_http(self):
            target = self._viser_target_path()
            try:
                n = int(self.headers.get("Content-Length", "0") or "0")
                body = self.rfile.read(n) if n > 0 else None
                hdrs = {k: v for k, v in self.headers.items()
                        if k.lower() not in ("host", "connection", "content-length")}
                conn = http.client.HTTPConnection("127.0.0.1", viser_port, timeout=30)
                conn.request(self.command, target, body=body, headers=hdrs)
                resp = conn.getresponse()
                data = resp.read()
                self.send_response(resp.status)
                skip = {"transfer-encoding", "connection", "content-length"}
                for k, v in resp.getheaders():
                    if k.lower() in skip:
                        continue
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                if data:
                    self.wfile.write(data)
            except Exception:
                try: self._send(502)
                except Exception: pass

        def _proxy_ws(self):
            """Bridge a WebSocket upgrade to viser by raw-socket forwarding.

            After we forward the upgrade request, viser's 101 response and all
            subsequent frames are shuttled both directions until either side
            closes."""
            try:
                up = socket.create_connection(("127.0.0.1", viser_port))
            except Exception:
                self._send(502); return
            head = [f"{self.command} {self._viser_target_path()} HTTP/1.1"]
            for k, v in self.headers.items():
                if k.lower() == "host":
                    head.append(f"Host: 127.0.0.1:{viser_port}")
                else:
                    head.append(f"{k}: {v}")
            up.sendall(("\r\n".join(head) + "\r\n\r\n").encode("latin-1"))
            client = self.connection
            def pipe(a, b):
                try:
                    while True:
                        buf = a.recv(8192)
                        if not buf: break
                        b.sendall(buf)
                except Exception:
                    pass
                try: b.shutdown(socket.SHUT_WR)
                except Exception: pass
            t1 = threading.Thread(target=pipe, args=(client, up), daemon=True)
            t2 = threading.Thread(target=pipe, args=(up, client), daemon=True)
            t1.start(); t2.start(); t1.join(); t2.join()
            try: up.close()
            except Exception: pass

        def _bright_param(self):
            """Read ``b=<float>`` from the query string, clamped to [1.0, 4.0]."""
            parts = self.path.split("?", 1)
            if len(parts) < 2:
                return 1.0
            for pair in parts[1].split("&"):
                if pair.startswith("b="):
                    try: return max(1.0, min(4.0, float(pair[2:])))
                    except ValueError: return 1.0
            return 1.0

        def _width_param(self):
            """Read ``w=<int>`` (bbox outline width in px), clamped to [1, 12]."""
            parts = self.path.split("?", 1)
            if len(parts) < 2:
                return 3
            for pair in parts[1].split("&"):
                if pair.startswith("w="):
                    try: return max(1, min(12, int(float(pair[2:]))))
                    except ValueError: return 3
            return 3

        def _q(self, key, default=None):
            """Read a single query-string parameter."""
            parts = self.path.split("?", 1)
            if len(parts) < 2:
                return default
            from urllib.parse import parse_qs
            vals = parse_qs(parts[1]).get(key)
            return vals[0] if vals else default

        def _serve_file_range(self, path, ctype):
            """Serve a file, honouring a single HTTP Range request (browsers need
            206/Range support to play/seek <video>)."""
            try:
                size = os.path.getsize(path)
            except OSError:
                self._send(404, b"not found"); return
            rng = self.headers.get("Range", "")
            if rng.startswith("bytes="):
                try:
                    a, _, b = rng[len("bytes="):].partition("-")
                    start = int(a) if a else 0
                    end = int(b) if b else size - 1
                    end = min(end, size - 1)
                    if start > end or start >= size:
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{size}")
                        self.end_headers(); return
                    length = end - start + 1
                    with open(path, "rb") as f:
                        f.seek(start); data = f.read(length)
                    self.send_response(206)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Length", str(length))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(data); return
                except Exception:
                    pass
            with open(path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.headers.get("Upgrade", "").lower() == "websocket":
                self._proxy_ws(); return
            p = self.path.split("?", 1)[0]
            if p == "/" or p == "/index.html":
                self._send(200, wrapper_html_filled.encode("utf-8"), "text/html; charset=utf-8")
            elif p.startswith("/img/tile_bg/"):
                # Pure splat-BG tile (no bbox overlay) — the client draws bboxes
                # on a <canvas> layered above the <img>. Result is cached by
                # (render_version, idx, brightness), so revisits cost ~0.
                try:
                    idx = int(p[len("/img/tile_bg/"):].split(".")[0])
                    assert 0 <= idx < len(state["keyframes"])
                except Exception:
                    self._send(400); return
                with state_lock:
                    blank = idx >= 1 and not camera_path_exists()
                if blank:
                    # No camera trajectory yet: keyframe tiles stay black (the
                    # panel itself and the input-image tile show as usual).
                    self._send(200, jpeg_bytes(np.zeros((tile_h, tile_w, 3), np.uint8),
                                               quality=88), "image/jpeg")
                    return
                self._send(200, tile_bg_jpeg(idx, self._bright_param()),
                           "image/jpeg")
            elif p == "/img/input_image.jpg":
                # The raw first video frame (frame 0) = the I2V input image, shown
                # as the non-selectable top row of the right panel.
                with state_lock:
                    inp = cv2.resize(frames[0], (tile_w, tile_h),
                                     interpolation=cv2.INTER_AREA)
                self._send(200, jpeg_bytes(inp, quality=88), "image/jpeg")
            elif p.startswith("/img/tile/"):
                # Legacy server-side full-render path (BG + bboxes baked in).
                # Kept for backwards compatibility; the live UI uses tile_bg.
                try:
                    idx = int(p[len("/img/tile/"):].split(".")[0])
                    assert 0 <= idx < len(state["keyframes"])
                except Exception:
                    self._send(400); return
                br = self._bright_param()
                bw = self._width_param()
                with state_lock:
                    img = render_single_tile(idx, bright=br, bbox_width=bw)
                self._send(200, jpeg_bytes(img), "image/jpeg")
            elif p == "/img/canvas.jpg":
                with state_lock:
                    img = render_canvas_now()
                self._send(200, jpeg_bytes(img, quality=88), "image/jpeg")
            elif p == "/img/canvas_raw.jpg":
                # Editor canvas — renders the ACTIVE keyframe's frame (the one
                # selected on the right-side tiles), from its edited pose, so the
                # editor background matches that tile. Falls back to the GT mp4
                # frame only when no point cloud is loaded.
                with state_lock:
                    fi = active_frame_idx()
                    c2w = state["c2w_edit"][fi]
                    K = intr[fi]
                    radius = int(state.get("splat_radius", 2))
                if pcd is None:
                    base = cv2.resize(frames[fi], (canvas_w, canvas_h),
                                      interpolation=cv2.INTER_AREA)
                else:
                    base = splat_render(pcd["points"], pcd["colors"],
                                        c2w, K,
                                        full_hw=(H, W),
                                        render_hw=(canvas_h, canvas_w),
                                        radius=radius)
                base = apply_brightness(base, self._bright_param())
                self._send(200, jpeg_bytes(base, quality=92), "image/jpeg")
            elif p == "/api/pcd_preview":
                if pcd is None:
                    self._send(404, b"no pcd", "text/plain"); return
                with state_lock:
                    pts = np.asarray(pcd["points"], dtype=np.float32)
                    cols = np.asarray(pcd["colors"], dtype=np.uint8)
                    cap = 52_000
                    if len(pts) > cap:
                        sel = np.random.default_rng(0).choice(len(pts), cap, replace=False)
                        pts = pts[sel]
                        cols = cols[sel]
                    n = int(len(pts))
                    blob = bytearray(4 + n * 12 + n * 3)
                    struct.pack_into("<I", blob, 0, n)
                    blob[4:4 + n * 12] = pts.tobytes()
                    blob[4 + n * 12:4 + n * 12 + n * 3] = cols.tobytes()
                self._send(200, bytes(blob), "application/octet-stream")
            elif p == "/api/fps_init":
                with state_lock:
                    kfi = int(state["active_kf_idx"])
                    fi = state["keyframes"][max(0, min(kfi, len(state["keyframes"]) - 1))]
                    c2w = state["c2w_edit"][fi]
                    pos = c2w[:3, 3].astype(float)
                    fwd = c2w[:3, 2].astype(float)
                    fov_deg = float(np.degrees(fov_y_from_intrinsic(intr[fi], H)))
                out = {
                    "position": pos.tolist(),
                    "fwd": fwd.tolist(),
                    "fov_y_deg": fov_deg,
                    "has_pcd": pcd is not None,
                }
                self._send(200, json.dumps(out).encode("utf-8"), "application/json")
            elif p == "/api/state":
                # Per-frame layout model: objects (identity) + boxes (each tagged
                # with obj_id + frame). Tiles draw each frame's own 2D boxes —
                # no cross-frame projection — so the camera matrices are no longer
                # needed client-side. Client polls render_version for BG refresh.
                with state_lock:
                    kfi = max(0, min(int(state["active_kf_idx"]),
                                     len(render_view_indices) - 1))
                    data = {
                        "W": int(W), "H": int(H), "N": int(N),
                        "target_frame": state["target_frame"],
                        "objects": [dict(o) for o in state["objects"]],
                        "boxes": [dict(b) for b in state["boxes"]],
                        "current_obj_id": state["current_obj_id"],
                        "render_version": state["render_version"],
                        "tile_w": int(tile_w), "tile_h": int(tile_h),
                        "render_view_indices": list(render_view_indices),
                        # Which keyframe/frame the editor is bound to + palette
                        # for the tile borders / 3D-frustum pairing.
                        "active_kf_idx": kfi,
                        "active_frame": int(render_view_indices[kfi]),
                        "keyframe_colors": [list(palette_color(i))
                                            for i in range(len(render_view_indices))],
                        "caption": caption,
                        "gen_load_status": gen_state["load_status"],
                        "gen_load_msg": gen_state["load_msg"],
                        "preloaded": bool(preloaded),
                        "has_scene": bool(has_scene),
                        "default_seed": int(default_seed),
                        "moge_load_status": moge_state["load_status"],
                        "moge_load_msg": moge_state["load_msg"],
                        "pcd_point_size": float(state["pcd_point_size"]),
                    }
                self._send(200, json.dumps(data).encode("utf-8"),
                           "application/json")
            elif p == "/api/generate/status":
                job_id = self._q("job")
                resp = {
                    "load_status": gen_state["load_status"],
                    "load_msg": gen_state["load_msg"],
                }
                job = gen_jobs.get(job_id) if job_id else None
                if job is not None:
                    resp.update({
                        "job_id": job_id,
                        "status": job["status"],
                        "progress": job.get("progress", 0.0),
                        "msg": job.get("msg", ""),
                        "seed": job.get("seed"),
                        "error": job.get("error"),
                        "video": (f"/api/generate/video?job={job_id}"
                                  if job["status"] == "done" else None),
                        "video_annotated": (
                            f"/api/generate/video_annotated?job={job_id}"
                            if job["status"] == "done" and job.get("video_annotated")
                            else None),
                    })
                self._send(200, json.dumps(resp).encode("utf-8"), "application/json")
            elif p == "/api/generate/video":
                job_id = self._q("job")
                job = gen_jobs.get(job_id) if job_id else None
                if (job is None or job.get("status") != "done"
                        or not job.get("video")
                        or not os.path.exists(job["video"])):
                    self._send(404, b"not ready"); return
                self._serve_file_range(job["video"], "video/mp4")
            elif p == "/api/generate/video_annotated":
                job_id = self._q("job")
                job = gen_jobs.get(job_id) if job_id else None
                if (job is None or job.get("status") != "done"
                        or not job.get("video_annotated")
                        or not os.path.exists(job["video_annotated"])):
                    self._send(404, b"not ready"); return
                self._serve_file_range(job["video_annotated"], "video/mp4")
            else:
                self._proxy_http()

        def do_POST(self):
            p = self.path.split("?", 1)[0]
            if p == "/api/object":
                obj = add_object()
                self._send(200, json.dumps(obj).encode(), "application/json")
            elif p == "/api/box":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                    obj_id = int(body["obj_id"]); frame = int(body["frame"])
                except Exception:
                    self._send(400); return
                coords = {k: body[k] for k in
                          ("x1", "y1", "x2", "y2", "front_depth", "thickness") if k in body}
                box = add_box(obj_id, frame, coords)
                if box is None:
                    self._send(404, b'{"ok":false}', "application/json"); return
                self._send(200, json.dumps(box).encode(), "application/json")
            elif p == "/api/box/carry":
                # Carry an object onto `frame` from its nearest placed frame,
                # preserving WORLD position (reprojected into this frame's camera).
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                    obj_id = int(body["obj_id"]); frame = int(body["frame"])
                except Exception:
                    self._send(400); return
                box = carry_box(obj_id, frame)
                if box is None:
                    self._send(404, b'{"ok":false}', "application/json"); return
                self._send(200, json.dumps(box).encode(), "application/json")
            elif p.startswith("/api/object/") and p.endswith("/current"):
                try:
                    oid = int(p[len("/api/object/"):-len("/current")])
                except Exception:
                    self._send(400); return
                ok = set_current_obj(oid)
                self._send(200 if ok else 404,
                           json.dumps({"ok": ok}).encode(), "application/json")
            elif p.startswith("/api/object/") and p.endswith("/rename"):
                try:
                    oid = int(p[len("/api/object/"):-len("/rename")])
                    length = int(self.headers.get("Content-Length", "0"))
                    body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                except Exception:
                    self._send(400); return
                ok = rename_object(oid, body.get("name", ""))
                self._send(200 if ok else 404,
                           json.dumps({"ok": ok}).encode(), "application/json")
            elif p == "/api/reset":
                reset_session()
                self._send(200, json.dumps({"ok": True}).encode(), "application/json")
            elif p == "/api/save_layout":
                msg = save_layout()
                print(f"[viewer] {msg}")
                self._send(200, json.dumps({"msg": msg}).encode(), "application/json")
            elif p == "/api/active_kf":
                # Tile click → make that keyframe active (mirror of the 3D
                # frustum click, which calls set_active_kf directly).
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                    idx = int(body.get("idx", 0))
                except Exception:
                    self._send(400); return
                with state_lock:
                    set_active_kf(idx)
                self._send(200, b'{"ok":true}', "application/json")
            elif p == "/api/traj_fps_apply":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    body = (json.loads(self.rfile.read(length).decode("utf-8"))
                            if length else {})
                except Exception:
                    self._send(400); return
                caps = body.get("captures")
                orient = body.get("orient", "tangent")
                orient_mode = "look_at_scene" if orient == "look_at" else "tangent"
                with state_lock:
                    msg = apply_fps_trajectory_from_captures(caps or [], orient_mode)
                    ok = msg.startswith("set trajectory")
                    if ok:
                        full_rebuild()
                self._send(200, json.dumps({"ok": ok, "msg": msg}).encode("utf-8"),
                           "application/json")
            elif p == "/api/upload_image":
                if moge_state["load_status"] != "ready" or moge_state["runner"] is None:
                    self._send(503, json.dumps({
                        "ok": False,
                        "msg": moge_state.get("load_msg", "MoGe not ready"),
                        "load_status": moge_state["load_status"],
                    }).encode(), "application/json")
                    return
                if not moge_lock.acquire(blocking=False):
                    self._send(409, json.dumps({
                        "ok": False, "msg": "an upload is already processing",
                    }).encode(), "application/json")
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    body = (json.loads(self.rfile.read(length).decode("utf-8"))
                            if length else {})
                    data_url = body.get("image", "")
                    if "," in data_url:
                        data_url = data_url.split(",", 1)[1]
                    raw = base64.b64decode(data_url)
                    img = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
                    camera = None
                    if body.get("camera_npz"):
                        z = np.load(io.BytesIO(base64.b64decode(body["camera_npz"])))
                        camera = {"extrinsic": z["extrinsic"], "intrinsic": z["intrinsic"]}
                    layout = body.get("layout")
                    if isinstance(layout, str):
                        layout = json.loads(layout) if layout.strip() else None
                    if layout and "instances" not in layout and "frames" in layout:
                        # the editor's own layout_edited.json: take its last annotated frame
                        fr = layout["frames"]
                        boxes = fr[max(fr, key=lambda k: int(k))] if fr else []
                        names = {o["id"]: o.get("name") for o in layout.get("objects", [])}
                        layout = {"instances": [{"bbox": b["bbox_2d"], "caption": b.get("name") or names.get(b["obj_id"])}
                                                for b in boxes]}
                    res = _handle_upload(img, prompt=body.get("prompt", ""), camera=camera, layout=layout)
                    self._send(200, json.dumps(res).encode(), "application/json")
                except Exception as e:
                    traceback.print_exc()
                    self._send(500, json.dumps({"ok": False, "msg": str(e)}).encode(),
                               "application/json")
                finally:
                    try: moge_lock.release()
                    except RuntimeError: pass
            elif p == "/api/generate":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    body = (json.loads(self.rfile.read(length).decode("utf-8"))
                            if length else {})
                except Exception:
                    self._send(400); return
                if gen_state["load_status"] != "ready" or gen_state["gen"] is None:
                    self._send(503, json.dumps({
                        "ok": False,
                        "msg": gen_state.get("load_msg", "model not ready"),
                        "load_status": gen_state["load_status"],
                    }).encode(), "application/json")
                    return
                if not gen_lock.acquire(blocking=False):
                    self._send(409, json.dumps({
                        "ok": False, "msg": "a generation is already running",
                    }).encode(), "application/json")
                    return
                try:
                    seed_mode = str(body.get("seed_mode", "default"))
                    if seed_mode == "custom":
                        try: seed = int(body.get("seed"))
                        except Exception: seed = int(default_seed)
                    elif seed_mode == "random":
                        seed = random.randint(0, 2**31 - 1)
                    else:
                        seed = int(default_seed)
                    try: steps = max(1, min(100, int(body.get("steps", 50))))
                    except Exception: steps = 50
                    try: cfg = float(body.get("cfg", 6.0))
                    except Exception: cfg = 6.0
                    job_id = uuid.uuid4().hex[:12]
                    params = {"seed": seed, "prompt": body.get("prompt"),
                              "steps": steps, "cfg": cfg, "num_frames": 81,
                              "height": 352, "width": 640, "fps": 16}
                    gen_jobs[job_id] = {"status": "queued", "progress": 0.0,
                                        "msg": "queued", "video": None,
                                        "error": None, "seed": seed}
                    threading.Thread(target=_run_generation,
                                     args=(job_id, params), daemon=True).start()
                except Exception as e:
                    try: gen_lock.release()
                    except RuntimeError: pass
                    self._send(500, json.dumps({"ok": False, "msg": str(e)}).encode(),
                               "application/json")
                    return
                self._send(200, json.dumps({
                    "ok": True, "job_id": job_id, "seed": seed,
                }).encode(), "application/json")
            else:
                self._proxy_http()

        def do_PATCH(self):
            p = self.path.split("?", 1)[0]
            if p.startswith("/api/box/"):
                try:
                    box_id = int(p[len("/api/box/"):])
                    length = int(self.headers.get("Content-Length", "0"))
                    body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                except Exception:
                    self._send(400); return
                ok = update_box(box_id, body)
                self._send(200 if ok else 404, json.dumps({"ok": ok}).encode(),
                           "application/json")
            else:
                self._proxy_http()

        def do_DELETE(self):
            p = self.path.split("?", 1)[0]
            if p.startswith("/api/box/"):
                try:
                    box_id = int(p[len("/api/box/"):])
                except Exception:
                    self._send(400); return
                remove_box(box_id)
                self._send(200, b'{"ok":true}', "application/json")
            elif p.startswith("/api/object/"):
                try:
                    oid = int(p[len("/api/object/"):])
                except Exception:
                    self._send(400); return
                remove_object(oid)
                self._send(200, b'{"ok":true}', "application/json")
            else:
                self._proxy_http()

    httpd = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"[viewer] open  http://localhost:{args.port}/")

    server.sleep_forever()


if __name__ == "__main__":
    main()
