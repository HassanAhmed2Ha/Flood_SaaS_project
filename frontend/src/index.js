import * as THREE from "three";
import { OrbitControls } from 'jsm/controls/OrbitControls.js';
import getStarfield from "./getStarfield.js";
import { getFresnelMat } from "./getFresnelMat.js";
import { getEarthMat } from "./getEarthMat.js";
import getLayer from "./getLayer.js";
const w = window.innerWidth;
const h = window.innerHeight;
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, w / h, 0.1, 1000);
camera.position.set(0.2, 0, 3);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(w, h);
document.body.appendChild(renderer.domElement);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.outputColorSpace = THREE.LinearSRGBColorSpace;

// const loader = new THREE.TextureLoader();

const sunDirection = new THREE.Vector3(-2, 0.5, 1.5);

const earthGroup = new THREE.Group();
earthGroup.rotation.z = -23.4 * Math.PI / 180;
scene.add(earthGroup);
const ctrls = new OrbitControls(camera, renderer.domElement);
ctrls.enableDamping = true;

const detail = 32;

const geometry = new THREE.IcosahedronGeometry(1, detail);
const material = getEarthMat(sunDirection);
const earthMesh = new THREE.Mesh(geometry, material);
earthGroup.add(earthMesh);

const atmosphereMat = getFresnelMat();
const glowMesh = new THREE.Mesh(geometry, atmosphereMat);
glowMesh.scale.setScalar(1.02);
earthGroup.add(glowMesh);

// add twinkle
const stars = getStarfield({ numStars: 2000 });
scene.add(stars);

const sunLight = new THREE.DirectionalLight(0xffffff, 4.0);
sunLight.position.copy(sunDirection);
scene.add(sunLight);
const sunGeo = new THREE.SphereGeometry(1, 32, 32);
const sunMat = new THREE.MeshBasicMaterial({ color: 0xffff00 });
const sunMesh = new THREE.Mesh(sunGeo, sunMat);
sunMesh.position.copy(sunDirection).multiplyScalar(5);
scene.add(sunMesh);

const nebula = getLayer({ path: './textures/rad-grad.png' });
scene.add(nebula);

// ==========================================
// ANIMATION LOOP
// ==========================================

// Globe rotation speed — slows during transition
let globeRotationSpeed = 0.002;

function animate(t = 0) {
  requestAnimationFrame(animate);
  earthMesh.rotation.y += globeRotationSpeed;
  glowMesh.rotation.y += globeRotationSpeed;
  stars.userData.update(t);
  renderer.render(scene, camera);
  ctrls.update();
}

animate();

function handleWindowResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);

  if (tacticalMap) {
    tacticalMap.invalidateSize();
  }
}
window.addEventListener('resize', handleWindowResize, false);

// ==========================================
// PHASE 5: DYNAMIC ZOOM-TO-MAP TRANSITION
// ==========================================

// --- DOM refs ---
const scanBtn = document.getElementById('scan-btn');
const statusBox = document.getElementById('status-box');
const mapContainer = document.getElementById('map2d');
const returnBtn = document.getElementById('return-globe-btn');
const mapHud = document.getElementById('map-hud');
const vignette = document.getElementById('zoom-vignette');

// --- State ---
let tacticalMap = null;
let floodLayer = null;
let isScanning = false;
let isTransitioning = false;
let savedCameraPos = new THREE.Vector3();
let savedCameraTarget = new THREE.Vector3();

// ==========================================
// COORDINATE CONVERSION
// ==========================================

/**
 * Convert geographic lat/lon to a 3D position on the globe surface.
 * Accounts for the earthGroup's axial tilt (-23.4°).
 * Returns a Vector3 at the given altitude above the surface.
 */
function latLonToGlobePosition(lat, lon, altitude = 1.6) {
  const phi = (90 - lat) * (Math.PI / 180);    // Polar angle from north pole
  const theta = (lon + 180) * (Math.PI / 180);  // Azimuthal angle

  // Position on unit sphere
  const x = -Math.sin(phi) * Math.cos(theta);
  const y = Math.cos(phi);
  const z = Math.sin(phi) * Math.sin(theta);

  const surfacePoint = new THREE.Vector3(x, y, z);

  // Apply the earthGroup's axial tilt rotation
  const tiltQuat = new THREE.Quaternion();
  tiltQuat.setFromEuler(earthGroup.rotation);
  surfacePoint.applyQuaternion(tiltQuat);

  // Account for the current rotation of the earth mesh
  const rotQuat = new THREE.Quaternion();
  rotQuat.setFromAxisAngle(new THREE.Vector3(0, 1, 0), earthMesh.rotation.y);
  surfacePoint.applyQuaternion(rotQuat);

  // Scale to altitude above surface (1.0 = surface, 1.6 = 0.6 above)
  return surfacePoint.multiplyScalar(altitude);
}

// ==========================================
// EASING & INTERPOLATION
// ==========================================

/** Smooth cubic ease-in-out */
function easeInOutCubic(t) {
  return t < 0.5
    ? 4 * t * t * t
    : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

/** Lerp a value */
function lerp(a, b, t) {
  return a + (b - a) * t;
}

// ==========================================
// LEAFLET MAP (SINGLETON)
// ==========================================

function initTacticalMap(lat, lon) {
  if (tacticalMap) {
    tacticalMap.setView([lat, lon], 13);
    return tacticalMap;
  }

  tacticalMap = L.map('map2d', {
    center: [lat, lon],
    zoom: 13,
    zoomControl: true,
    attributionControl: true,
  });

  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://carto.com/">CARTO</a> | Flood Intelligence AI',
    subdomains: 'abcd',
    maxZoom: 19
  }).addTo(tacticalMap);

  return tacticalMap;
}

function renderFloodExtent(geojsonData) {
  if (floodLayer) {
    tacticalMap.removeLayer(floodLayer);
    floodLayer = null;
  }

  if (!geojsonData || !geojsonData.features || geojsonData.features.length === 0) {
    console.warn('[TACTICAL] No flood features to render.');
    return;
  }

  floodLayer = L.geoJSON(geojsonData, {
    style: function () {
      return {
        color: '#00aaff',
        weight: 2,
        opacity: 0.9,
        fillColor: '#0066cc',
        fillOpacity: 0.35,
        className: 'flood-polygon'
      };
    },
    onEachFeature: function (feature, layer) {
      const props = feature.properties || {};
      const area = props.area_sqkm ? `${props.area_sqkm.toFixed(2)} km²` : 'N/A';
      layer.bindPopup(
        `<div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#fff;background:rgba(10,15,30,0.95);padding:12px;border-radius:8px;border:1px solid rgba(0,170,255,0.3);">
           <strong style="color:#00aaff;letter-spacing:1px;">FLOOD ZONE</strong><br/>
           <span style="color:rgba(255,255,255,0.6);">Area:</span> ${area}
         </div>`,
        { className: 'tactical-popup', closeButton: false }
      );
    }
  }).addTo(tacticalMap);

  const bounds = floodLayer.getBounds();
  if (bounds.isValid()) {
    tacticalMap.fitBounds(bounds, { padding: [60, 60] });
  }
}

// ==========================================
// ZOOM-TO-MAP TRANSITION (3D → 2D)
// ==========================================

/**
 * Cinematic zoom-to-map transition.
 * 
 * Timeline (2000ms total):
 *   0%–100%  : Camera interpolates from current → top-down view above target
 *   0%–100%  : Canvas opacity fades 1.0 → 0.15
 *   0%–60%   : Vignette fades in 0 → 1
 *   60%–100% : Vignette holds at 1
 *   80%–100% : Leaflet map fades in 0 → 1 (crossfade overlap)
 *   100%     : Show HUD, show Return button, trigger scan-line
 */
function transitionToTacticalMap(lat, lon, data) {
  if (isTransitioning) return;
  isTransitioning = true;

  // Save current camera state for return animation
  savedCameraPos.copy(camera.position);
  savedCameraTarget.copy(ctrls.target);

  // Disable orbit controls during transition
  ctrls.enabled = false;

  // Freeze earth rotation
  globeRotationSpeed = 0;

  // Compute target camera position: top-down above the lat/lon
  const targetPos = latLonToGlobePosition(lat, lon, 1.6);

  // The camera will look toward the globe center (origin)
  const lookTarget = new THREE.Vector3(0, 0, 0);

  // Pre-initialize Leaflet map (hidden)
  initTacticalMap(lat, lon);
  mapContainer.classList.add('active');
  mapContainer.style.opacity = '0';

  // Animation config
  const duration = 2000; // 2 seconds
  const startTime = performance.now();
  const startPos = camera.position.clone();
  const startFov = camera.fov;
  const targetFov = 35; // Narrow FOV = magnification effect

  function zoomInFrame(now) {
    const elapsed = now - startTime;
    const rawT = Math.min(elapsed / duration, 1.0);
    const t = easeInOutCubic(rawT);

    // --- Camera position: lerp toward target ---
    camera.position.lerpVectors(startPos, targetPos, t);
    camera.lookAt(lookTarget);

    // --- FOV: narrow for magnification effect ---
    camera.fov = lerp(startFov, targetFov, t);
    camera.updateProjectionMatrix();

    // --- Canvas opacity: 1.0 → 0.15 ---
    const canvasOpacity = lerp(1.0, 0.15, t);
    renderer.domElement.style.opacity = canvasOpacity;

    // --- Vignette: fade in during first 60%, hold ---
    const vignetteT = Math.min(rawT / 0.6, 1.0);
    vignette.style.opacity = easeInOutCubic(vignetteT);

    // --- Leaflet map: fade in during last 20% (80%–100%) ---
    if (rawT >= 0.8) {
      const mapT = (rawT - 0.8) / 0.2; // 0→1 over the last 20%
      mapContainer.style.opacity = easeInOutCubic(mapT);
    }

    // --- Globe rotation speed: decelerate ---
    globeRotationSpeed = lerp(0.002, 0, Math.min(t * 2, 1));

    if (rawT < 1.0) {
      requestAnimationFrame(zoomInFrame);
    } else {
      // --- Animation complete ---
      renderer.domElement.classList.add('dimmed');
      mapContainer.style.opacity = '1';

      // Trigger scan-line effect
      mapContainer.classList.add('scan-active');
      setTimeout(() => mapContainer.classList.remove('scan-active'), 2200);

      // Ensure tiles render at correct size
      setTimeout(() => tacticalMap.invalidateSize(), 100);

      // Render flood data
      if (data.geojson) {
        renderFloodExtent(data.geojson);
      }

      // Update HUD
      document.getElementById('hud-confidence').textContent = `${data.confidence_score}%`;
      // Dynamic gauge arc — circumference = 2π × r(42) ≈ 263.9
      const gaugeArc = document.getElementById('gauge-arc');
      if (gaugeArc) {
        const circumference = 2 * Math.PI * 42;
        const pct = parseFloat(data.confidence_score) / 100;
        gaugeArc.style.strokeDashoffset = circumference * (1 - pct);
      }
      document.getElementById('hud-area').textContent = `${data.metrics.total_flood_area_sqkm} km²`;
      document.getElementById('hud-buildings').textContent = data.metrics.buildings_damaged;
      document.getElementById('hud-roads').textContent = data.metrics.roads_damaged_km ?? '—';
      document.getElementById('hud-farmland').textContent = data.metrics.farmland_damaged_sqkm
        ? `${data.metrics.farmland_damaged_sqkm} km²`
        : '—';

      // Show HUD and Return button
      mapHud.classList.add('visible');
      returnBtn.classList.add('visible');

      isTransitioning = false;
    }
  }

  requestAnimationFrame(zoomInFrame);
}

// ==========================================
// ZOOM-OUT TRANSITION (2D → 3D)
// ==========================================

/**
 * Reverse transition: zoom out from tactical map back to the 3D globe.
 * 
 * Timeline (1500ms total):
 *   0%–40%   : Leaflet map fades out 1 → 0
 *   0%–100%  : Camera interpolates back to saved position
 *   0%–100%  : Canvas opacity fades 0.15 → 1.0
 *   40%–100% : Vignette fades out 1 → 0
 *   100%     : Hide map, re-enable controls, restore rotation
 */
function transitionToGlobe() {
  if (isTransitioning) return;
  isTransitioning = true;

  // Hide HUD and return button immediately
  mapHud.classList.remove('visible');
  returnBtn.classList.remove('visible');

  const duration = 1500;
  const startTime = performance.now();
  const startPos = camera.position.clone();
  const startFov = camera.fov;
  const targetFov = 75; // Original FOV

  function zoomOutFrame(now) {
    const elapsed = now - startTime;
    const rawT = Math.min(elapsed / duration, 1.0);
    const t = easeInOutCubic(rawT);

    // --- Leaflet map: fade out during first 40% ---
    if (rawT <= 0.4) {
      const mapT = rawT / 0.4;
      mapContainer.style.opacity = lerp(1, 0, easeInOutCubic(mapT));
    } else {
      mapContainer.style.opacity = '0';
    }

    // --- Camera: interpolate back to saved position ---
    camera.position.lerpVectors(startPos, savedCameraPos, t);
    camera.lookAt(ctrls.target);

    // --- FOV: restore ---
    camera.fov = lerp(startFov, targetFov, t);
    camera.updateProjectionMatrix();

    // --- Canvas opacity: 0.15 → 1.0 ---
    const canvasOpacity = lerp(0.15, 1.0, t);
    renderer.domElement.style.opacity = canvasOpacity;

    // --- Vignette: fade out during 40%–100% ---
    if (rawT >= 0.4) {
      const vigT = (rawT - 0.4) / 0.6;
      vignette.style.opacity = lerp(1, 0, easeInOutCubic(vigT));
    }

    // --- Globe rotation: gradually restore ---
    globeRotationSpeed = lerp(0, 0.002, t);

    if (rawT < 1.0) {
      requestAnimationFrame(zoomOutFrame);
    } else {
      // --- Animation complete ---
      renderer.domElement.classList.remove('dimmed');
      renderer.domElement.style.opacity = '1';
      mapContainer.classList.remove('active');
      mapContainer.style.opacity = '0';
      vignette.style.opacity = '0';

      // Re-enable orbit controls
      ctrls.enabled = true;
      ctrls.target.copy(savedCameraTarget);
      ctrls.update();

      // Restore rotation
      globeRotationSpeed = 0.002;

      // Reset scan button
      scanBtn.innerText = "INITIATE SATELLITE SCAN";
      scanBtn.style.background = "";
      statusBox.style.display = "none";

      isTransitioning = false;
    }
  }

  requestAnimationFrame(zoomOutFrame);
}

// Return to Globe button handler
returnBtn.addEventListener('click', transitionToGlobe);

// ==========================================
// SCAN BUTTON EVENT LISTENER
// ==========================================

scanBtn.addEventListener('click', async () => {
  if (isTransitioning) return;

  // 1. Read input values
  const lat = parseFloat(document.getElementById('lat').value);
  const lon = parseFloat(document.getElementById('lon').value);
  const start_date = document.getElementById('start_date').value;
  const end_date = document.getElementById('end_date').value;
  const radius_km = parseInt(document.getElementById('scan_radius').value, 10);

  // 2. Update button state
  scanBtn.disabled = true;
  scanBtn.innerText = "SCANNING IN PROGRESS...";
  scanBtn.style.background = "#444";
  statusBox.style.display = "block";

  // 3. Status log — initial message
  if (radius_km > 0) {
    statusBox.innerHTML = `<span style="color:#ffcc00;">[WIDE AREA SCAN — ${radius_km}km radius]</span><br>Target: ${lat}, ${lon}`;
  } else {
    statusBox.innerHTML = `[UPLINK ESTABLISHED]<br>Target: ${lat}, ${lon}`;
  }

  // 4. Simulated Tactical Progress Stream
  const progressMessages = [
    '[SYS] Establishing secure uplink to Earth Engine...',
    '[NET] Generating regional grid matrix based on radius...',
    '[CPU] Allocating concurrent worker threads...',
    '[SAT] Fetching Sentinel SAR/Optical multi-band data...',
    '[AI]  U-Net model active. Ingesting satellite telemetry...',
    '[TILE] Processing geospatial quadrants. Standby...',
    '[SYS] Mosaicking arrays. Normalizing topological vectors...',
    '[AI]  Phase 1 Complete. Extracting flood polygons...',
  ];
  let msgIndex = 0;
  const progressInterval = setInterval(() => {
    if (msgIndex < progressMessages.length) {
      const msg = progressMessages[msgIndex];
      statusBox.innerHTML += `<br><span style="color:#00ccff;opacity:0.85;">${msg}</span>`;
      // Auto-scroll to bottom
      statusBox.scrollTop = statusBox.scrollHeight;
      msgIndex++;
    } else {
      // All messages exhausted — show a looping indicator
      statusBox.innerHTML += `<br><span style="color:#00ccff;opacity:0.5;">⟳ Awaiting engine response...</span>`;
      statusBox.scrollTop = statusBox.scrollHeight;
    }
  }, 3500);

  // 5. Cinematic camera zoom (subtle pre-scan approach)
  isScanning = true;

  // 6. Call the Node.js API Gateway
  try {
    const response = await fetch('https://flood-api-gateway.hassanahmed07-e9.workers.dev/api/scan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        latitude: lat,
        longitude: lon,
        start_date: start_date,
        end_date: end_date,
        radius_km: radius_km,
        tile_size_km: 5,
        max_workers: 4
      })
    });

    const data = await response.json();

    if (response.ok && data.task_id) {
      const task_id = data.task_id;
      // Stop the initial quick progress interval as we transition to polling
      clearInterval(progressInterval);

      statusBox.innerHTML = `[SYS] Task scheduled.<br>Task ID: ${task_id}<br><span style="color:#00ccff;opacity:0.85;">[SYS] Processing large grid. Polling for results...</span>`;
      statusBox.scrollTop = statusBox.scrollHeight;

      // Start polling status
      const pollingInterval = setInterval(async () => {
        try {
          const statusRes = await fetch(`https://flood-api-gateway.hassanahmed07-e9.workers.dev/api/status/${task_id}`);
          if (!statusRes.ok) {
            throw new Error(`HTTP Error ${statusRes.status}`);
          }
          const taskState = await statusRes.json();

          if (taskState.status === "completed") {
            clearInterval(pollingInterval);
            const finalData = taskState.data;

            statusBox.innerHTML = `[SUCCESS] AI Confidence: ${finalData.confidence_score}%<br>
                                   Flood Area: ${finalData.metrics.total_flood_area_sqkm} sq km<br>
                                   Buildings Damaged: ${finalData.metrics.buildings_damaged}<br>
                                   <span style="color:#00aaff;font-size:11px;">Initiating tactical zoom...</span>`;
            statusBox.style.color = "#fff";
            statusBox.style.borderLeftColor = "#00aaff";
            statusBox.style.background = "rgba(0, 170, 255, 0.15)";
            statusBox.scrollTop = statusBox.scrollHeight;

            scanBtn.innerText = "VIEW TACTICAL MAP";
            scanBtn.style.background = "linear-gradient(135deg, #0055cc, #0088ff)";
            scanBtn.disabled = false;
            isScanning = false;

            // Brief pause to show success, then begin the cinematic zoom
            setTimeout(() => {
              transitionToTacticalMap(lat, lon, finalData);
            }, 800);

          } else if (taskState.status === "failed") {
            clearInterval(pollingInterval);
            statusBox.innerHTML = `[ERROR] Task execution failed.<br><span style="color:#ff4444;font-size:11px;">${taskState.error || 'Unknown error'}</span>`;
            statusBox.style.borderLeftColor = "red";
            statusBox.style.background = "rgba(255, 0, 0, 0.1)";
            statusBox.scrollTop = statusBox.scrollHeight;
            scanBtn.innerText = "SCAN FAILED — RETRY";
            scanBtn.style.background = "linear-gradient(135deg, #cc2200, #ff4444)";
            scanBtn.disabled = false;
            isScanning = false;
          } else {
            // Task is still "processing"
            statusBox.innerHTML = `[SYS] Task ID: ${task_id}<br><span style="color:#00ccff;opacity:0.85;">[SYS] Processing large grid. Polling for results...</span>`;
            statusBox.scrollTop = statusBox.scrollHeight;
          }
        } catch (pollErr) {
          // If polling fails temporarily (network issue), keep polling or show error
          console.warn("[Polling] Error checking task status:", pollErr.message);
          statusBox.innerHTML += `<br><span style="color:#ffcc00;font-size:11px;">[WARN] Retrying connection: ${pollErr.message}</span>`;
          statusBox.scrollTop = statusBox.scrollHeight;
        }
      }, 5000); // Poll every 5 seconds

    } else {
      clearInterval(progressInterval);
      const errMsg = data.detail || data.error || "Failed to schedule task.";
      statusBox.innerHTML = `[ERROR] ${errMsg}`;
      statusBox.style.borderLeftColor = "red";
      statusBox.style.background = "rgba(255, 0, 0, 0.1)";
      statusBox.scrollTop = statusBox.scrollHeight;
      scanBtn.innerText = "SCAN FAILED — RETRY";
      scanBtn.style.background = "linear-gradient(135deg, #cc2200, #ff4444)";
      scanBtn.disabled = false;
      isScanning = false;
    }
  } catch (error) {
    clearInterval(progressInterval);
    statusBox.innerHTML = `[CRITICAL ERROR] Connection to Gateway lost.<br><span style="color:#ff4444;font-size:11px;">${error.message}</span>`;
    statusBox.style.borderLeftColor = "red";
    statusBox.style.background = "rgba(255, 0, 0, 0.1)";
    statusBox.scrollTop = statusBox.scrollHeight;
    scanBtn.innerText = "RECONNECT";
    scanBtn.disabled = false;
    isScanning = false;
  }
});

// ==========================================
// INTEL DOSSIER — Dynamic Component Loader
// ==========================================

(async function loadDossierComponent() {
  try {
    const res = await fetch('about_component.html');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const html = await res.text();

    // Inject into DOM
    const wrapper = document.createElement('div');
    wrapper.innerHTML = html;
    document.body.appendChild(wrapper);

    // Bind event listeners on dynamically loaded elements
    const aboutBtnEl = document.getElementById('about-btn');
    const overlay = document.getElementById('about-overlay');
    const closeBtn = document.getElementById('about-close');

    if (aboutBtnEl && overlay && closeBtn) {
      aboutBtnEl.addEventListener('click', () => {
        overlay.classList.add('active');
      });

      closeBtn.addEventListener('click', () => {
        overlay.classList.remove('active');
      });

      overlay.addEventListener('click', (e) => {
        if (e.target === overlay) {
          overlay.classList.remove('active');
        }
      });

      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && overlay.classList.contains('active')) {
          overlay.classList.remove('active');
        }
      });
    }

    console.log('[DOSSIER] Intel component loaded successfully.');
  } catch (err) {
    console.warn('[DOSSIER] Failed to load component:', err.message);
  }
})();