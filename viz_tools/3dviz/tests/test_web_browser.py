"""Opt-in browser regression with an actual local server and WebGL renderer.

Run with ``THREEDVIZ_BROWSER_TESTS=1`` and Playwright installed. Set
``THREEDVIZ_BROWSER_BINARY`` when Chrome/Chromium is not on PATH. No browser
download or external server is needed.
"""

from __future__ import annotations

import base64
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.parse import parse_qs, urlsplit
from urllib.request import urlopen

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("THREEDVIZ_BROWSER_TESTS") != "1",
    reason="Set THREEDVIZ_BROWSER_TESTS=1 with Playwright and Chrome/Chromium",
)


@pytest.fixture
def browser_viewer(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    executable = os.environ.get("THREEDVIZ_BROWSER_BINARY") or next(
        (path for name in ("google-chrome", "chromium", "chromium-browser")
         if (path := shutil.which(name))), None,
    )
    if not executable:
        pytest.skip("Chrome/Chromium is unavailable")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    with (tmp_path / "server.log").open("w+") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "threedviz", "--demo", "--no-open-browser", "--port", str(port)],
            stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy(),
        )
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    log.seek(0)
                    pytest.fail(f"Local viewer exited during startup:\n{log.read()}")
                try:
                    with urlopen(f"{base_url}/api/bootstrap", timeout=1) as response:
                        if response.status == 200:
                            break
                except (URLError, TimeoutError):
                    time.sleep(.05)
            else:
                log.seek(0)
                pytest.fail(f"Local viewer did not start:\n{log.read()}")
            with playwright.sync_playwright() as driver:
                browser = driver.chromium.launch(
                    executable_path=executable, headless=True,
                    args=["--no-sandbox", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
                )
                context = browser.new_context(
                    viewport={"width": 1440, "height": 1000},
                    permissions=["clipboard-read", "clipboard-write"],
                )
                page = context.new_page()
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
                failed = []
                page.on("requestfailed", lambda request: failed.append(f"{request.url}: {request.failure}"))
                try:
                    yield page, base_url, errors, failed
                finally:
                    context.close()
                    browser.close()
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _camera(page):
    return json.loads(page.locator("#camera-json").input_value())


def _wait_camera_change(page, previous):
    page.wait_for_function(
        "old => JSON.stringify(JSON.parse(document.querySelector('#camera-json').value)) !== JSON.stringify(old)",
        arg=previous,
    )


def _input(page, selector, value):
    page.locator(selector).evaluate(
        "(element, value) => {element.value=value; element.dispatchEvent(new Event('input', {bubbles:true}));}",
        str(value),
    )


def _change_number(page, selector, value):
    field = page.locator(selector)
    field.fill(str(value))
    field.press("Tab")


def _idle(page):
    page.wait_for_function("document.querySelector('#app').getAttribute('aria-busy') === 'false'")


def _rendered_region(page):
    # Exclude the floating controls so style labels cannot create a false pass.
    page.wait_for_timeout(120)
    return page.screenshot(
        clip={"x": 470, "y": 360, "width": 900, "height": 460},
        style="#camera-panel {visibility:hidden !important;}",
    )


def test_browser_faceted_points_keep_pixel_size(browser_viewer):
    page, url, errors, failed = browser_viewer
    fixture = {"count": 1}

    def point_packet(route):
        count = fixture["count"]
        revision = int(parse_qs(urlsplit(route.request.url).query)["revision"][0])
        header = {
            "protocol": 1, "revision": revision, "count": count, "origin": [0, 0, 0],
            "bounds": {"min": [-1, -1, -1], "max": [1, 1, 1]},
            "frame": {"id": "glyph", "scene_id": "glyph", "index": 0, "timestamp_us": None},
            "source_id": "browser-fixture", "reference_frame_id": "glyph",
            "annotation_status": "available", "boxes": [], "rgb_count": 0,
            "display": {"rgb": False, "raw": False},
            "buffers": [
                {"name": "positions", "dtype": "<f4", "shape": [count, 3], "offset": 0, "length": 12 * count},
                {"name": "distance", "dtype": "<f4", "shape": [count], "offset": 12 * count, "length": 4 * count},
            ],
        }
        encoded = json.dumps(header).encode()
        encoded += b" " * (-len(encoded) % 4)
        payload = struct.pack("<4f", 0, 0, 0, 40) if count else b""
        route.fulfill(content_type="application/octet-stream", body=struct.pack("<I", len(encoded)) + encoded + payload)

    # Observe actual GPU submissions without exposing renderer internals in the app.
    page.add_init_script("""(() => {
        window.__glyphDraws = [];
        for (const name of ['drawArraysInstanced', 'drawElementsInstanced']) {
            const original = WebGL2RenderingContext.prototype[name];
            WebGL2RenderingContext.prototype[name] = function(...args) {
                window.__glyphDraws.push({mode: args[0], count: args[name === 'drawArraysInstanced' ? 2 : 1],
                    instances: args[name === 'drawArraysInstanced' ? 3 : 4]});
                return original.apply(this, args);
            };
        }
    })();""")
    page.route("**/api/sessions/*/frame.bin?*", point_packet)

    def prepare_marker():
        page.goto(url)
        page.locator("#frame-stats").filter(has_text="1 points").wait_for()
        _idle(page)
        page.locator("details").filter(has=page.locator("#axes")).locator("summary").click()
        page.locator("#axes").uncheck()
        page.evaluate("window.__glyphDraws = []")
        _input(page, "#point-size", 12)

    def marker_pixels():
        page.wait_for_timeout(120)
        viewport = page.evaluate("({width: innerWidth, height: innerHeight})")
        image = page.screenshot(clip={"x": viewport["width"] / 2 - 16, "y": viewport["height"] / 2 - 16,
                                      "width": 32, "height": 32})
        decoded = page.evaluate("""async encoded => {
            const image = new Image();
            image.src = 'data:image/png;base64,' + encoded;
            await image.decode();
            const surface = document.createElement('canvas');
            surface.width = image.width; surface.height = image.height;
            const context = surface.getContext('2d');
            context.drawImage(image, 0, 0);
            return {width: image.width, pixels: [...context.getImageData(0, 0, image.width, image.height).data]};
        }""", base64.b64encode(image).decode())
        width, pixels = decoded["width"], decoded["pixels"]
        colors = {(index % width, index // width): tuple(pixels[index * 4:index * 4 + 3])
                  for index in range(len(pixels) // 4) if max(pixels[index * 4:index * 4 + 3]) < 230}
        if not colors:
            return None
        left, right = min(x for x, _ in colors), max(x for x, _ in colors)
        top, bottom = min(y for _, y in colors), max(y for _, y in colors)
        interior = Counter(color for (x, y), color in colors.items()
                           if all((x + dx, y + dy) in colors for dx in (-1, 0, 1) for dy in (-1, 0, 1)))
        capture_scale = width / 32
        return {"width": (right - left + 1) / capture_scale,
                "height": (bottom - top + 1) / capture_scale,
                "coverage": len(colors) / ((right - left + 1) * (bottom - top + 1)),
                "face_shades": sum(amount >= 3 for amount in interior.values())}

    prepare_marker()
    initial = marker_pixels()
    assert initial is not None
    assert 8 <= initial["width"] <= 13 and 8 <= initial["height"] <= 13
    assert .5 < initial["coverage"] < .92, "The marker must have a filled polygonal silhouette rather than a square"
    draws = page.evaluate("window.__glyphDraws")
    assert any(draw["mode"] == 4 and draw["count"] / 3 == 20 and draw["instances"] == 1 for draw in draws)

    farther = _camera(page)
    farther["position"] = [value * 2 for value in farther["position"]]
    page.locator("#camera-json").fill(json.dumps(farther))
    page.locator("#apply-camera").click()
    page.set_viewport_size({"width": 1000, "height": 700})
    resized = marker_pixels()
    assert resized is not None
    for dimension in ("width", "height"):
        assert resized[dimension] == pytest.approx(initial[dimension], abs=1)

    farther["fov"] = 179
    page.locator("#camera-json").fill(json.dumps(farther))
    page.locator("#apply-camera").click()
    wide_angle = marker_pixels()
    assert wide_angle is not None, "A supported wide camera FOV must not hide the point"
    for dimension in ("width", "height"):
        assert wide_angle[dimension] == pytest.approx(initial[dimension], abs=1)

    cdp = page.context.new_cdp_session(page)
    cdp.send("Emulation.setDeviceMetricsOverride", {"width": 1000, "height": 700, "deviceScaleFactor": 2, "mobile": False})
    prepare_marker()
    assert page.evaluate("devicePixelRatio") == 2
    assert page.locator("#viewport canvas").evaluate("element => element.height") == 1400
    high_dpi = marker_pixels()
    assert high_dpi is not None
    assert high_dpi["face_shades"] >= 3, "A filled marker must show distinct flat face shading"
    for dimension in ("width", "height"):
        assert high_dpi[dimension] == pytest.approx(initial[dimension], abs=1)

    fixture["count"] = 0
    page.locator("#next").click()
    _idle(page)
    assert " ".join(page.locator("#frame-stats").inner_text().split()) == "0 points"
    assert marker_pixels() is None, "An empty frame must remove the previous point"
    assert not errors, errors
    assert not failed, failed


def test_browser_loading_suffix_cycles_and_stops(browser_viewer):
    page, url, errors, failed = browser_viewer
    held_poll = []

    def hold_first_poll(route):
        if not held_poll:
            held_poll.append(route)
        else:
            route.continue_()

    # Keep a real request pending while inspecting the loading animation.
    page.route("**/api/sessions/*/jobs/*", hold_first_poll)
    page.goto(url)
    page.locator("#load-strip").wait_for(state="visible")
    deadline = time.monotonic() + 5
    while not held_poll and time.monotonic() < deadline:
        page.wait_for_timeout(20)
    assert held_poll, "The initial dataset request must remain pending during animation checks"
    page.evaluate("document.fonts.ready")

    dots = page.locator("#status-footer .loading-dots")
    assert page.locator(".status-dot").count() == 0
    assert dots.is_visible()
    assert dots.text_content() == "..."
    assert dots.get_attribute("aria-hidden") == "true"
    assert dots.evaluate("element => element.previousElementSibling.id") == "status"
    assert not page.locator("#status").inner_text().endswith(("…", "..."))
    assert dots.evaluate("element => element.getAnimations().some(animation => animation.playState === 'running')")
    announcement = page.locator("#announcements").inner_text()
    assert not page.locator("#announcements").evaluate(
        "element => element.contains(document.querySelector('.loading-dots'))"
    )

    # Sample animation phases directly so slow machines cannot miss a dot state.
    clip_paths, rendered_states = [], []
    for phase in (.05, .38, .72, 1.05):
        clip_paths.append(dots.evaluate("""async (element, phase) => {
            const animation = element.getAnimations()[0];
            animation.pause();
            await animation.ready;
            animation.currentTime = animation.effect.getComputedTiming().duration * phase;
            await new Promise(resolve => requestAnimationFrame(resolve));
            return getComputedStyle(element).clipPath;
        }""", phase))
        rendered_states.append(dots.screenshot())
        assert page.locator("#announcements").inner_text() == announcement
    assert len(set(clip_paths[:3])) == 3
    assert clip_paths[3] == clip_paths[0], "The loading suffix must wrap to its first state"
    assert len(set(rendered_states[:3])) == 3, "Each dot state must produce a visible change"
    assert rendered_states[3] == rendered_states[0]

    held_poll[0].continue_()
    _idle(page)
    assert dots.is_hidden()
    assert dots.evaluate("element => element.getAnimations().length") == 0
    assert page.locator("#status-footer").is_hidden()

    page.locator("#dataset").select_option("nuscenes")
    page.locator("#root").fill("/does-not-exist/3dviz-loading-regression")
    page.locator("#version").fill("v1.0-mini")
    page.locator("#connect").click()
    page.locator("#error").wait_for(state="visible")
    _idle(page)
    assert page.locator("#status-footer").is_visible()
    assert dots.is_hidden()
    assert dots.evaluate("element => element.getAnimations().length") == 0
    assert not errors, errors
    assert not failed, failed


def test_browser_demo_navigation_styling_camera_and_recovery(browser_viewer):
    page, url, errors, failed = browser_viewer
    requested_urls = []
    page.on("request", lambda request: requested_urls.append(request.url))
    # Hold the polling response briefly so the loading strip can be observed
    # even when the deterministic demo finishes in a few milliseconds.
    def slow_poll(route):
        time.sleep(.2)
        route.continue_()

    page.route("**/api/sessions/*/jobs/*", slow_poll)
    page.goto(url)
    page.locator("#load-strip").wait_for(state="visible")
    page.locator("#frame-stats").filter(has_text="30,000").wait_for()
    _idle(page)
    assert page.locator("#status-footer").is_hidden()
    announcement = page.locator("#announcements")
    assert page.locator("#app #announcements").count() == 0
    assert announcement.get_attribute("role") == "status"
    assert "Frame 1 of 36" in announcement.inner_text()
    assert "loaded" in announcement.inner_text().lower()
    assert page.locator("#box-section").is_hidden()
    assert page.locator("#scene option").count() == 2
    assert page.locator("#point-colors option").all_text_contents() == ["Gray", "Viridis", "Camera RGB", "Concerto"]
    assert page.locator('#point-colors option[value="rgb"]').is_disabled()
    assert page.locator("#raw-points").is_disabled()
    canvas = page.locator("#viewport canvas")
    assert canvas.bounding_box() == {"x": 0, "y": 0, "width": 1440, "height": 1000}
    assert page.locator(".navbar").evaluate("element => getComputedStyle(element).position") == "fixed"
    navbar = page.locator(".navbar").bounding_box()
    assert navbar["height"] < 60, "Desktop dataset navigation should remain compact"
    brand_logo = page.locator(".brand img.brand-logo")
    page.wait_for_function("""() => {
        const logo = document.querySelector('.brand img.brand-logo');
        return logo && logo.complete && logo.naturalWidth > 0;
    }""")
    logo_origin = urlsplit(brand_logo.evaluate("element => element.currentSrc"))
    viewer_origin = urlsplit(url)
    assert (logo_origin.scheme, logo_origin.netloc) == (viewer_origin.scheme, viewer_origin.netloc)
    assert page.locator(".brand svg").count() == 0
    assert page.locator("#sidebar").evaluate("element => getComputedStyle(element).position") == "fixed"
    assert page.get_by_role("heading", name="Scene controls", exact=True).count() == 1
    assert page.get_by_text("YOUR WORKSPACE", exact=True).count() == 0
    assert page.locator(".live-badge").count() == 0
    assert page.locator("#sidebar #camera-json").count() == 0
    assert page.locator("#root").is_hidden()
    assert page.locator("#scene-description").is_visible()
    assert page.locator("#scene-description").inner_text().strip()
    assert page.locator("#camera-hint").is_hidden()
    loaded_fonts = page.evaluate("""async () => {
        await document.fonts.ready;
        return [...document.fonts].filter(face => face.status === 'loaded').map(face => face.family);
    }""")
    assert "Source Sans Pro" in [family.strip("'\"") for family in loaded_fonts]
    camera_panel = page.locator("#camera-panel")
    assert camera_panel.evaluate("element => element.open")
    camera_box = camera_panel.bounding_box()
    assert camera_box["x"] > 1440 / 2
    assert camera_box["y"] >= navbar["y"] + navbar["height"]
    assert camera_box["x"] + camera_box["width"] <= 1440

    styles = []
    for scheme in ("gray", "viridis", "concerto"):
        page.locator("#point-colors").select_option(scheme)
        styles.append(_rendered_region(page))
    assert len(set(styles)) == 3, "The selected distance palettes must change the rendered points"
    before = _rendered_region(page)
    _input(page, "#point-size", 7)
    assert page.locator("#point-size-number").input_value() == "7"
    assert _rendered_region(page) != before
    before = _rendered_region(page)
    _change_number(page, "#point-size-number", 4.5)
    assert page.locator("#point-size").input_value() == "4.5"
    assert _rendered_region(page) != before
    before = _rendered_region(page)
    for invalid in (0, 13, 4.25, ""):
        _change_number(page, "#point-size-number", invalid)
        assert page.locator("#point-size-number").input_value() == "4.5"
        assert page.locator("#point-size").input_value() == "4.5"
    assert _rendered_region(page) == before, "Invalid entries must retain the current rendered point size"
    before = _rendered_region(page)
    _input(page, "#max-distance", 40)
    assert _rendered_region(page) != before
    before = _rendered_region(page)
    _change_number(page, "#max-distance-number", 100)
    assert page.locator("#max-distance").input_value() == "100"
    assert _rendered_region(page) != before
    for invalid in (0, 1001, 100.5, ""):
        _change_number(page, "#max-distance-number", invalid)
        assert page.locator("#max-distance-number").input_value() == "100"
        assert page.locator("#max-distance").input_value() == "100"

    detection_tab = page.get_by_role("tab", name="Detection")
    scene_tab = page.get_by_role("tab", name="Scene")
    detection_tab.click()
    assert page.locator("#box-section").is_visible()
    assert page.locator("#box-total").inner_text() == "8"
    car = page.get_by_role("checkbox", name="Show car boxes", exact=True)
    before = _rendered_region(page)
    car.uncheck()
    assert _rendered_region(page) != before
    car.check()
    before = _rendered_region(page)
    _input(page, "#box-width", 6)
    assert page.locator("#box-width-number").input_value() == "6"
    assert _rendered_region(page) != before
    before = _rendered_region(page)
    _change_number(page, "#box-width-number", 3.5)
    assert page.locator("#box-width").input_value() == "3.5"
    assert _rendered_region(page) != before
    before = _rendered_region(page)
    for invalid in (0, 11, 3.25, ""):
        _change_number(page, "#box-width-number", invalid)
        assert page.locator("#box-width-number").input_value() == "3.5"
        assert page.locator("#box-width").input_value() == "3.5"
    assert _rendered_region(page) == before, "Invalid entries must retain the current rendered line width"
    before = _rendered_region(page)
    _input(page, '[aria-label="car box color"]', "#00ffff")
    assert _rendered_region(page) != before
    page.locator("#hide-all").click()
    assert not page.locator("#classes input[type=checkbox]:checked").count()
    page.locator("#show-all").click()
    assert page.locator("#classes input[type=checkbox]:checked").count() == 10
    scene_tab.click()

    menu_toggle = page.locator("#menu-toggle")
    sidebar_content = page.locator("#sidebar-content")
    assert page.locator(".navbar #menu-toggle").count() == 0
    assert page.locator("#sidebar .panel-heading #menu-toggle").count() == 1
    saved_controls = {
        selector: page.locator(selector).input_value()
        for selector in ("#point-size", "#box-width", "#max-distance", "#point-colors", "#scene", "#frame-number")
    }
    saved_camera = _camera(page)
    # The panel header remains usable after minimizing, with mouse and keyboard.
    for activate in (menu_toggle.click, lambda: menu_toggle.press("Enter")):
        activate()
        assert menu_toggle.get_attribute("aria-expanded") == "false"
        assert sidebar_content.is_hidden()
        assert sidebar_content.evaluate("element => element.inert")
        assert page.locator("#point-size-number").is_hidden()
        assert menu_toggle.is_visible()
        assert page.get_by_role("heading", name="Scene controls", exact=True).is_visible()
        assert page.evaluate("document.activeElement.id") == "menu-toggle"
        activate()
        assert menu_toggle.get_attribute("aria-expanded") == "true"
        assert sidebar_content.is_visible()
        assert not sidebar_content.evaluate("element => element.inert")
        assert page.locator("#point-size-number").is_visible()
        for selector, value in saved_controls.items():
            assert page.locator(selector).input_value() == value
        assert _camera(page) == saved_camera

    # Loading settings can be collapsed without hiding temporal navigation.
    page.locator("#source-section summary").click()
    frame_slider = page.locator("#frame-slider")
    assert frame_slider.is_visible()
    assert frame_slider.is_enabled()
    frame_slider.focus()
    frame_slider.press("ArrowRight")
    _idle(page)
    assert page.locator("#frame-number").input_value() == "2"
    frame_slider.press("ArrowLeft")
    _idle(page)
    assert page.locator("#frame-number").input_value() == "1"

    view_details = page.locator("details").filter(has=page.locator("#axes"))
    view_details.locator("summary").click()
    initial = _camera(page)
    page.mouse.move(1000, 420)
    page.mouse.down()
    page.mouse.move(1100, 475, steps=10)
    page.mouse.up()
    _wait_camera_change(page, initial)
    orbit = _camera(page)
    page.mouse.wheel(0, -250)
    _wait_camera_change(page, orbit)
    zoom = _camera(page)
    page.mouse.move(1050, 450)
    page.mouse.down(button="right")
    page.mouse.move(1110, 450, steps=8)
    page.mouse.up(button="right")
    page.wait_for_function(
        "pivot => JSON.stringify(JSON.parse(document.querySelector('#camera-json').value).pivot) !== JSON.stringify(pivot)",
        arg=zoom["pivot"],
    )
    assert _camera(page)["pivot"] != zoom["pivot"]

    view = {"position": [18., -26., 11.], "rotation": [23., -17., 41.],
            "pivot": [2.5, -1.25, 3.], "fov": 43., "near": .35, "far": 650.}
    draft = json.dumps(view, indent=2)
    page.locator("#camera-json").fill(draft)
    assert page.locator("#camera-hint").is_visible()
    before = _rendered_region(page)
    page.mouse.move(1000, 420)
    page.mouse.down()
    page.mouse.move(1060, 455, steps=8)
    page.mouse.up()
    assert _rendered_region(page) != before
    assert page.locator("#camera-json").input_value() == draft, "Orbiting must preserve an unapplied camera draft"
    page.locator("#copy-camera").click()
    page.wait_for_function("document.querySelector('#camera-hint').textContent === 'Camera view copied.'")
    assert json.loads(page.evaluate("navigator.clipboard.readText()")) == view
    assert page.locator("#camera-json").input_value() == draft, "Copy must preserve the visible camera draft"
    page.mouse.wheel(0, -100)
    page.wait_for_timeout(120)
    assert page.locator("#camera-json").input_value() == draft
    page.locator("#apply-camera").click()
    assert page.locator("#camera-hint").is_visible()
    for key, values in view.items():
        assert _camera(page)[key] == pytest.approx(values, abs=1e-7)
    page.locator("#copy-camera").click()
    page.wait_for_function("document.querySelector('#camera-hint').textContent === 'Camera view copied.'")
    assert page.locator("#camera-hint").is_visible()
    assert json.loads(page.evaluate("navigator.clipboard.readText()")) == _camera(page)
    page.locator("#camera-json").fill("{invalid}")
    page.locator("#apply-camera").click()
    assert page.locator("#camera-hint").is_visible()
    assert page.locator("#camera-json").get_attribute("aria-invalid") == "true"
    assert page.locator("#camera-hint").inner_text() != "Camera view applied."
    invalid_draft = page.locator("#camera-json").input_value()
    page.evaluate("text => navigator.clipboard.writeText(text)", "{invalid clipboard}")
    page.locator("#paste-camera").click()
    page.wait_for_function("document.querySelector('#camera-hint').textContent.startsWith('Clipboard:')")
    assert page.locator("#camera-json").input_value() == invalid_draft
    assert page.locator("#camera-json").get_attribute("aria-invalid") == "true"

    pasted_view = {**view, "position": [-15., 19., 14.], "fov": 51.}
    before = _rendered_region(page)
    page.evaluate("text => navigator.clipboard.writeText(text)", json.dumps(pasted_view))
    page.locator("#paste-camera").click()
    page.wait_for_function("document.querySelector('#camera-hint').textContent.startsWith('View pasted')")
    assert _camera(page) == pasted_view
    assert page.locator("#camera-json").get_attribute("aria-invalid") != "true"
    assert page.locator("#camera-hint").is_visible()
    assert _rendered_region(page) == before, "Pasting a camera draft must wait for Apply"
    page.locator("#apply-camera").click()
    for key, values in pasted_view.items():
        assert _camera(page)[key] == pytest.approx(values, abs=1e-7)
    assert _rendered_region(page) != before

    page.locator("#reset-camera").click()
    reset = _camera(page)
    assert reset["position"] != pasted_view["position"]
    assert page.locator("#camera-hint").is_hidden()
    assert page.locator("#camera-json").get_attribute("aria-invalid") != "true"
    # The axes toggle must not alter the camera or become screen-spanning.
    before = _rendered_region(page)
    page.locator("#axes").uncheck()
    page.locator("#axes").check()
    page.wait_for_timeout(120)
    assert _camera(page) == reset
    assert _rendered_region(page) == before

    page.locator("#next").click()
    _idle(page)
    assert page.locator("#frame-number").input_value() == "2"
    for key, values in reset.items():
        assert _camera(page)[key] == pytest.approx(values, abs=1e-7), "Adjacent frames should preserve the current camera"
    page.locator("#frame-number").fill("36")
    page.locator("#frame-number").press("Enter")
    _idle(page)
    assert page.locator("#next").is_disabled()
    page.locator("#scene").select_option("demo-highway")
    _idle(page)
    assert page.locator("#frame-number").input_value() == "1"
    detection_tab.click()
    assert page.locator("#box-total").inner_text() == "7"
    scene_tab.click()
    page.locator("#scene").select_option("demo-urban")
    _idle(page)
    assert "30,000" in page.locator("#frame-stats").inner_text()

    # A bad new source must leave the successfully loaded source and its view
    # available, including mouse interaction while the error is displayed.
    if not page.locator("#source-section").evaluate("element => element.open"):
        page.locator("#source-section summary").click()
    saved = _camera(page)
    before = _rendered_region(page)
    page.locator("#dataset").select_option("nuscenes")
    page.locator("#root").fill("/does-not-exist/3dviz-browser-regression")
    page.locator("#version").fill("v1.0-mini")
    page.locator("#connect").click()
    page.locator("#error").wait_for(state="visible")
    _idle(page)
    assert page.locator("#dataset").input_value() == "demo"
    assert page.locator("#scene").input_value() == "demo-urban"
    assert _camera(page) == saved
    page.locator("#dismiss-error").click()
    assert _rendered_region(page) == before

    _input(page, "#point-size", 3)
    _input(page, "#max-distance", 80)
    detection_tab.click()
    _input(page, "#box-width", 2)
    scene_tab.click()
    page.locator("#point-colors").select_option("concerto")
    view_details.locator("summary").click()
    page.locator("#sidebar").evaluate("element => {element.scrollTop=0;}")
    screenshot = os.environ.get("THREEDVIZ_BROWSER_SCREENSHOT")
    if screenshot:
        page.screenshot(path=str(Path(screenshot)), full_page=True)
    page.set_viewport_size({"width": 390, "height": 844})
    page.wait_for_function("""() => {
        const {width, height} = document.querySelector('#viewport canvas').getBoundingClientRect();
        return width === 390 && height === 844;
    }""")
    assert canvas.bounding_box() == {"x": 0, "y": 0, "width": 390, "height": 844}
    assert page.evaluate("document.documentElement.scrollWidth") == 390
    assert not camera_panel.evaluate("element => element.open")
    camera_panel.locator("summary").click()
    assert page.locator("#camera-json").is_visible()
    narrow_camera = camera_panel.bounding_box()
    narrow_nav = page.locator(".navbar").bounding_box()
    assert narrow_camera["y"] >= narrow_nav["y"] + narrow_nav["height"]
    assert 0 <= narrow_camera["x"] < narrow_camera["x"] + narrow_camera["width"] <= 390
    # The native details toggle event updates the sidebar on a later task.
    page.wait_for_function("document.querySelector('#menu-toggle').getAttribute('aria-expanded') === 'false'")
    assert page.locator("#menu-toggle").get_attribute("aria-expanded") == "false"
    assert sidebar_content.is_hidden()
    assert sidebar_content.evaluate("element => element.inert")
    assert menu_toggle.is_visible()
    page.locator("#menu-toggle").click()
    assert page.locator("#menu-toggle").get_attribute("aria-expanded") == "true"
    assert sidebar_content.is_visible()
    assert not sidebar_content.evaluate("element => element.inert")
    assert not camera_panel.evaluate("element => element.open")
    page.locator("#menu-toggle").click()
    assert page.locator("#menu-toggle").get_attribute("aria-expanded") == "false"
    camera_panel.locator("summary").click()
    assert page.locator("#camera-json").is_visible()
    if screenshot:
        path = Path(screenshot)
        page.screenshot(path=str(path.with_name(path.stem.removesuffix("-demo") + "-narrow" + path.suffix)), full_page=True)
    origin = urlsplit(url)
    external_requests = [
        requested for requested in requested_urls
        if (parts := urlsplit(requested)).scheme in {"http", "https"}
        and (parts.scheme, parts.netloc) != (origin.scheme, origin.netloc)
    ]
    assert not external_requests, "The viewer and its font must work without external network requests"
    assert any(
        urlsplit(requested).path.lower().endswith((".woff2", ".woff", ".ttf", ".otf"))
        for requested in requested_urls
    ), "Source Sans Pro must be loaded from the local viewer server"
    assert not errors, errors
    assert not failed, failed


def test_browser_detection_filters_preserve_the_point_cloud_and_camera(browser_viewer):
    page, url, errors, failed = browser_viewer
    frame_requests = []
    page.on("request", lambda request: frame_requests.append(request.url) if "/frame.bin?" in request.url else None)
    page.goto(url)
    page.locator("#frame-stats").filter(has_text="30,000").wait_for()
    _idle(page)
    boxes = page.locator("#box-section")
    detection_tab = page.get_by_role("tab", name="Detection")
    assert boxes.is_hidden()
    detection_tab.click()
    assert boxes.is_visible()
    score = page.locator("#confidence")
    assert score.is_enabled()
    camera = _camera(page)
    count = page.locator('#detection-counts [data-count="output"]')
    assert int(count.inner_text()) > 0
    assert page.locator("#detection-run-name").inner_text() == "Demo predictions"
    # The review pipeline receives the slider's committed threshold. No cloud job is needed.
    score.evaluate("element => {element.value='1';element.dispatchEvent(new Event('change'));}")
    _idle(page)
    assert count.inner_text() == "0"
    assert "removed by the current filters" in page.locator("#detection-status").inner_text()
    assert len(frame_requests) == 1

    page.locator("#advanced-detection > summary").click()
    page.locator("#reset-filters").click()
    _idle(page)
    assert int(count.inner_text()) > 0
    page.locator("#layer-predictions").uncheck()
    _input(page, "#box-width", 4)
    car = page.get_by_role("checkbox", name="Show car boxes", exact=True)
    car.uncheck()
    assert int(count.inner_text()) > 0
    page.get_by_role("tab", name="LiDAR seg").click()
    assert boxes.is_hidden()
    assert page.locator("#labels-ground-truth").is_disabled()
    page.get_by_role("tab", name="Scene").click()
    assert boxes.is_hidden()
    assert not page.locator("#layer-predictions").is_checked()
    assert page.locator("#layer-gt").is_checked()
    assert page.locator("#frame-number").input_value() == "1"
    detection_tab.click()
    assert boxes.is_visible()
    assert page.locator("#box-width-number").input_value() == "4"
    assert not car.is_checked()
    assert _camera(page) == camera
    assert len(frame_requests) == 1
    assert not errors, errors
    assert not failed, failed

