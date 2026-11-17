/**
 * AuraBurn AI — Interactive Neural Diagnostic Console
 * Orchestrates real-time Computer Vision segmentation & CNN inference UI
 */

document.addEventListener('DOMContentLoaded', () => {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const viewportCard = document.getElementById('viewport-card');
  const viewportImg = document.getElementById('viewport-img');
  const scannerSweep = document.getElementById('scanner-sweep');
  const emptyConsole = document.getElementById('empty-console');
  const activeConsole = document.getElementById('active-console');
  const sampleCards = document.querySelectorAll('.sample-card');
  const viewTabBtns = document.querySelectorAll('.view-tab-btn');
  const resetBtn = document.getElementById('reset-btn');
  const printBtn = document.getElementById('print-btn');

  // State
  let currentResult = null;
  let activeViewMode = 'original'; // 'original' | 'mask' | 'overlay'

  // =========================================================================
  // Drag & Drop Handlers
  // =========================================================================
  if (dropzone && fileInput) {
    ['dragenter', 'dragover'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.add('drag-active');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.remove('drag-active');
      });
    });

    dropzone.addEventListener('drop', (e) => {
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFileUpload(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
      }
    });
  }

  // =========================================================================
  // Sample Selection Handlers
  // =========================================================================
  sampleCards.forEach(card => {
    card.addEventListener('click', () => {
      const filename = card.getAttribute('data-filename');
      if (filename) {
        analyzeSample(filename);
      }
    });
  });

  // =========================================================================
  // Viewport Multi-View Switcher Tabs
  // =========================================================================
  viewTabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      viewTabBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeViewMode = btn.getAttribute('data-mode');
      updateViewportImage();
    });
  });

  function updateViewportImage() {
    if (!currentResult || !viewportImg) return;
    
    viewportImg.style.opacity = '0.3';
    setTimeout(() => {
      if (activeViewMode === 'original') {
        viewportImg.src = currentResult.original_url;
      } else if (activeViewMode === 'mask') {
        viewportImg.src = currentResult.mask_url || currentResult.original_url;
      } else if (activeViewMode === 'overlay') {
        viewportImg.src = currentResult.overlay_url || currentResult.original_url;
      }
      viewportImg.style.opacity = '1';
    }, 120);
  }

  // =========================================================================
  // File Upload Pipeline
  // =========================================================================
  async function handleFileUpload(file) {
    if (!file.type.match('image.*')) {
      alert('Format file tidak didukung. Harap unggah gambar (PNG, JPG, JPEG, WEBP).');
      return;
    }

    // Preview lokal instan
    const reader = new FileReader();
    reader.onload = (e) => {
      showViewportPreview(e.target.result, file.name);
    };
    reader.readAsDataURL(file);

    // Kirim ke server
    const formData = new FormData();
    formData.append('file', file);

    setAnalyzingState(true);

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Accept': 'application/json' },
        body: formData
      });

      const data = await response.json();
      if (data.status === 'success') {
        renderAnalysisResults(data);
      } else {
        alert(data.message || 'Gagal memproses gambar.');
      }
    } catch (err) {
      console.error(err);
      alert('Terjadi kesalahan koneksi ke server analisis.');
    } finally {
      setAnalyzingState(false);
    }
  }

  // =========================================================================
  // Sample Case Pipeline
  // =========================================================================
  async function analyzeSample(filename) {
    setAnalyzingState(true);
    showViewportPreview(`/uploads/${filename}`, filename);

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({ sample: filename })
      });

      const data = await response.json();
      if (data.status === 'success') {
        renderAnalysisResults(data);
      } else {
        alert(data.message || 'Gagal menganalisis sampel.');
      }
    } catch (err) {
      console.error(err);
      alert('Terjadi kesalahan jaringan.');
    } finally {
      setAnalyzingState(false);
    }
  }

  // =========================================================================
  // State & Rendering
  // =========================================================================
  function showViewportPreview(src, filename) {
    if (viewportCard) viewportCard.style.display = 'block';
    if (viewportImg) viewportImg.src = src;
    
    const hudFilename = document.getElementById('hud-filename');
    if (hudFilename) hudFilename.textContent = filename || 'image_input';

    const hudTimestamp = document.getElementById('hud-timestamp');
    if (hudTimestamp) {
      hudTimestamp.textContent = new Date().toLocaleTimeString('id-ID', { hour12: false });
    }
  }

  function setAnalyzingState(isAnalyzing) {
    if (scannerSweep) {
      scannerSweep.style.display = isAnalyzing ? 'block' : 'none';
    }
    const dropPrompt = document.querySelector('.dropzone-prompt');
    if (dropPrompt) {
      dropPrompt.textContent = isAnalyzing ? 'Menjalankan Analisis Saraf & Segmentasi...' : 'Tarik & Letakkan Foto Luka Bakar';
    }
  }

  function renderAnalysisResults(data) {
    currentResult = data;
    const cls = data.classification;
    const seg = data.segmentation;
    const clinical = cls.clinical;

    // Tampilkan panel aktif
    if (emptyConsole) emptyConsole.style.display = 'none';
    if (activeConsole) activeConsole.style.display = 'block';

    // HUD Meta
    const hudDims = document.getElementById('hud-dimensions');
    if (hudDims && seg) hudDims.textContent = seg.image_dimensions || '150 x 150 px';

    // Banner Diagnosa Utama
    const banner = document.getElementById('diagnosis-banner');
    const gradeBadge = document.getElementById('res-grade-badge');
    const severityPill = document.getElementById('res-severity-pill');
    const diagTitle = document.getElementById('res-title');
    const diagDepth = document.getElementById('res-depth');

    if (gradeBadge) {
      gradeBadge.textContent = clinical.grade;
      gradeBadge.style.backgroundColor = clinical.badge_color;
      gradeBadge.style.color = '#ffffff';
    }
    if (severityPill) severityPill.textContent = `Severity: ${clinical.severity}`;
    if (diagTitle) diagTitle.textContent = clinical.title;
    if (diagDepth) diagDepth.textContent = clinical.depth;

    if (banner) {
      banner.style.backgroundColor = `${clinical.badge_color}18`;
      banner.style.borderColor = `${clinical.badge_color}55`;
    }

    // Metrik Key
    const confVal = document.getElementById('res-confidence-val');
    if (confVal) {
      animateCounter(confVal, 0, cls.confidence, '%');
    }

    const burnAreaVal = document.getElementById('res-burnarea-val');
    if (burnAreaVal && seg) {
      burnAreaVal.textContent = `${seg.burn_percentage}%`;
    }

    // Distribusi Probabilitas 4 Kelas
    const probContainer = document.getElementById('probabilities-container');
    if (probContainer) {
      probContainer.innerHTML = '';
      cls.probabilities.forEach((item, index) => {
        const row = document.createElement('div');
        row.className = 'prob-row';
        row.innerHTML = `
          <div class="prob-meta">
            <span class="prob-name" style="${item.is_predicted ? 'color: #ffffff; font-weight: 700;' : ''}">${item.title}</span>
            <span class="prob-pct" style="${item.is_predicted ? `color: ${item.color};` : ''}">${item.probability}%</span>
          </div>
          <div class="prob-bar-track">
            <div class="prob-bar-fill" style="width: 0%; background: ${item.color};"></div>
          </div>
        `;
        probContainer.appendChild(row);

        // Staggered bar animation
        setTimeout(() => {
          const fill = row.querySelector('.prob-bar-fill');
          if (fill) fill.style.width = `${item.probability}%`;
        }, 80 + index * 60);
      });
    }

    // Protokol Klinis & Tindakan Pertama
    const protocolList = document.getElementById('protocol-list');
    if (protocolList && clinical.protocols) {
      protocolList.innerHTML = '';
      clinical.protocols.forEach(step => {
        const li = document.createElement('li');
        li.textContent = step;
        protocolList.appendChild(li);
      });
    }

    // Reset visual tab ke mode saat ini
    updateViewportImage();
  }

  function animateCounter(el, start, end, suffix = '') {
    const duration = 600;
    const startTime = performance.now();
    
    function update(time) {
      const elapsed = time - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // Easing out
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = (start + (end - start) * eased).toFixed(1);
      el.textContent = `${current}${suffix}`;
      
      if (progress < 1) {
        requestAnimationFrame(update);
      } else {
        el.textContent = `${end.toFixed(1)}${suffix}`;
      }
    }
    requestAnimationFrame(update);
  }

  // =========================================================================
  // Actions: Reset & Print
  // =========================================================================
  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      currentResult = null;
      if (viewportCard) viewportCard.style.display = 'none';
      if (activeConsole) activeConsole.style.display = 'none';
      if (emptyConsole) emptyConsole.style.display = 'flex';
      if (fileInput) fileInput.value = '';
    });
  }

  if (printBtn) {
    printBtn.addEventListener('click', () => {
      window.print();
    });
  }
});
