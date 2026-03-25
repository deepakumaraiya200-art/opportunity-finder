/* ============================================
   OPPORTUNITYFINDER — CLIENT-SIDE LOGIC
   Upload UX + Dashboard Filters
   ============================================ */

document.addEventListener('DOMContentLoaded', () => {

  // ---- MODE TOGGLE LOGIC ----
  const modeInternBtn = document.getElementById('modeIntern');
  const modeJobBtn = document.getElementById('modeJob');
  const modeSlider = document.getElementById('modeSlider');
  const searchTypeInput = document.getElementById('searchType');
  const countLabel = document.getElementById('countLabel');
  const lstepFindText = document.getElementById('lstepFindText');

  let currentMode = 'intern'; // default

  function setMode(mode) {
    currentMode = mode;
    if (searchTypeInput) searchTypeInput.value = mode;

    if (mode === 'job') {
      modeInternBtn.classList.remove('active');
      modeJobBtn.classList.add('active');
      if (modeSlider) modeSlider.classList.add('right');
      if (countLabel) countLabel.textContent = '📊 Number of jobs to show';
      if (lstepFindText) lstepFindText.textContent = 'Finding Jobs';
    } else {
      modeJobBtn.classList.remove('active');
      modeInternBtn.classList.add('active');
      if (modeSlider) modeSlider.classList.remove('right');
      if (countLabel) countLabel.textContent = '📊 Number of internships to show';
      if (lstepFindText) lstepFindText.textContent = 'Finding Internships';
    }
  }

  if (modeInternBtn) {
    modeInternBtn.addEventListener('click', () => setMode('intern'));
  }
  if (modeJobBtn) {
    modeJobBtn.addEventListener('click', () => setMode('job'));
  }

  // ---- UPLOAD PAGE LOGIC ----
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const uploadForm = document.getElementById('uploadForm');
  const uploadBtn = document.getElementById('uploadBtn');
  const filePreview = document.getElementById('filePreview');
  const fileName = document.getElementById('fileName');
  const fileSize = document.getElementById('fileSize');
  const removeFileBtn = document.getElementById('removeFile');
  const loadingOverlay = document.getElementById('loadingOverlay');

  if (dropZone && fileInput) {
    let selectedFile = null;

    // Drag & Drop
    ['dragenter', 'dragover'].forEach(event => {
      dropZone.addEventListener(event, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('drag-over');
      });
    });

    ['dragleave', 'drop'].forEach(event => {
      dropZone.addEventListener(event, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('drag-over');
      });
    });

    dropZone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files.length > 0) handleFile(files[0]);
    });

    dropZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', () => {
      if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
    });

    function handleFile(file) {
      if (file.type !== 'application/pdf') {
        showToast('Please upload a PDF file only.');
        dropZone.classList.add('shake');
        setTimeout(() => dropZone.classList.remove('shake'), 500);
        return;
      }

      if (file.size > 5 * 1024 * 1024) {
        showToast('File size must be under 5MB.');
        return;
      }

      selectedFile = file;
      fileName.textContent = file.name;
      fileSize.textContent = formatFileSize(file.size);
      filePreview.classList.add('visible');
      uploadBtn.disabled = false;
    }

    if (removeFileBtn) {
      removeFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        selectedFile = null;
        fileInput.value = '';
        filePreview.classList.remove('visible');
        uploadBtn.disabled = true;
      });
    }

    // Slider live counter
    const jobCountSlider = document.getElementById('jobCount');
    const countValue = document.getElementById('countValue');
    if (jobCountSlider && countValue) {
      jobCountSlider.addEventListener('input', () => {
        countValue.textContent = jobCountSlider.value;
      });
    }

    // ---- ROLE SELECTOR LOGIC ----
    const roleSelectorHeader = document.getElementById('roleSelectorHeader');
    const roleDropdown = document.getElementById('roleDropdown');
    const roleSelectorArrow = document.getElementById('roleSelectorArrow');
    const roleChosenLabel = document.getElementById('roleChosen');
    const targetRoleInput = document.getElementById('targetRole');
    const roleOtherInput = document.getElementById('roleOtherInput');
    const roleOtherBtn = document.getElementById('roleOtherBtn');

    if (roleSelectorHeader && roleDropdown) {
      // Toggle dropdown
      roleSelectorHeader.addEventListener('click', () => {
        roleDropdown.classList.toggle('open');
        roleSelectorArrow.classList.toggle('open');
      });

      // Track selected roles
      const selectedRoles = new Set();

      function updateRoleInput() {
        const rolesArray = [...selectedRoles];
        targetRoleInput.value = rolesArray.join(', ');
        if (rolesArray.length === 0) {
          roleChosenLabel.textContent = 'Select roles (optional)';
          roleChosenLabel.classList.remove('active');
        } else if (rolesArray.length === 1) {
          roleChosenLabel.textContent = rolesArray[0];
          roleChosenLabel.classList.add('active');
        } else {
          roleChosenLabel.textContent = `${rolesArray.length} roles selected`;
          roleChosenLabel.classList.add('active');
        }
      }

      // Chip toggle (multi-select)
      const allChips = document.querySelectorAll('.role-chip');
      allChips.forEach(chip => {
        chip.addEventListener('click', () => {
          const role = chip.dataset.role;
          if (chip.classList.contains('selected')) {
            chip.classList.remove('selected');
            selectedRoles.delete(role);
          } else {
            chip.classList.add('selected');
            selectedRoles.add(role);
          }
          updateRoleInput();
        });
      });

      // "Other" custom role — adds to selection
      if (roleOtherBtn && roleOtherInput) {
        roleOtherBtn.addEventListener('click', () => {
          const customRole = roleOtherInput.value.trim();
          if (!customRole) return;
          selectedRoles.add(customRole);
          roleOtherInput.value = '';
          updateRoleInput();
        });
        // Allow Enter key to submit custom role
        roleOtherInput.addEventListener('keydown', (e) => {
          if (e.key === 'Enter') {
            e.preventDefault();
            roleOtherBtn.click();
          }
        });
      }
    }

    if (uploadForm) {
      uploadForm.addEventListener('submit', (e) => {
        if (!selectedFile) {
          e.preventDefault();
          showToast('Please select a resume to upload.');
          return;
        }
        if (loadingOverlay) {
          loadingOverlay.classList.add('visible');
          animateLoadingSteps();
        }
      });
    }
  }

  // ---- DASHBOARD FILTER LOGIC ----
  const filterBtns = document.querySelectorAll('.filter-btn');
  const searchBox = document.getElementById('searchBox');

  if (filterBtns.length > 0) {
    filterBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        filterBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        applyFilters();
      });
    });
  }

  if (searchBox) {
    searchBox.addEventListener('input', () => applyFilters());
  }

  function applyFilters() {
    const activeBtn = document.querySelector('.filter-btn.active');
    const catFilter = activeBtn ? activeBtn.dataset.cat : 'all';
    const searchText = searchBox ? searchBox.value.toLowerCase().trim() : '';

    const allCards = document.querySelectorAll('.card');

    allCards.forEach(card => {
      const cardCats = (card.dataset.cat || '').toLowerCase();
      const cardText = card.textContent.toLowerCase();

      const matchesCat = catFilter === 'all' || cardCats.includes(catFilter);
      const matchesSearch = !searchText || cardText.includes(searchText);

      card.style.display = (matchesCat && matchesSearch) ? '' : 'none';
    });

    // Hide empty sections
    document.querySelectorAll('.grid').forEach(grid => {
      const visible = [...grid.querySelectorAll('.card')].some(c => c.style.display !== 'none');
      grid.style.display = visible ? '' : 'none';
    });

    document.querySelectorAll('.section-label').forEach(lbl => {
      const next = lbl.nextElementSibling;
      lbl.style.display = (next && next.style.display === 'none') ? 'none' : '';
    });

    // No results
    const anyVisible = document.querySelectorAll('.card:not([style*="display: none"])');
    const noResults = document.getElementById('noResults');
    if (noResults) {
      noResults.style.display = anyVisible.length === 0 ? 'block' : 'none';
    }
  }

  // ---- MATCH BAR ANIMATION ----
  const matchObserver = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const fill = entry.target.querySelector('.match-fill');
        if (fill) {
          const targetWidth = fill.style.width;
          fill.style.width = '0%';
          requestAnimationFrame(() => {
            requestAnimationFrame(() => {
              fill.style.width = targetWidth;
            });
          });
        }
        matchObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.3 });

  document.querySelectorAll('.match').forEach(m => matchObserver.observe(m));

  // ---- UTILITIES ----
  function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  }

  function showToast(message) {
    let toast = document.getElementById('toast');
    if (!toast) {
      toast = document.createElement('div');
      toast.id = 'toast';
      toast.className = 'toast';
      document.body.appendChild(toast);
    }
    toast.textContent = message;
    toast.classList.add('visible');
    setTimeout(() => toast.classList.remove('visible'), 3000);
  }

  function animateLoadingSteps() {
    const pctEl = document.getElementById('loadingPct');
    const fillEl = document.getElementById('progressFill');
    const titleEl = document.getElementById('loadingTitle');
    const subEl = document.getElementById('loadingSubtitle');
    const factEl = document.getElementById('loadingFact');

    const steps = [
      document.getElementById('lstep-0'),
      document.getElementById('lstep-1'),
      document.getElementById('lstep-2'),
      document.getElementById('lstep-3'),
    ];

    const modeLabel = currentMode === 'job' ? 'Jobs' : 'Internships';
    const modeLabel2 = currentMode === 'job' ? 'job' : 'internship';

    const phases = [
      { step: 0, title: 'Reading Your Resume', sub: 'Parsing PDF and extracting text', pctStart: 0, pctEnd: 8, duration: 2000 },
      { step: 1, title: 'Extracting Your Skills', sub: 'AI is analyzing your technical expertise', pctStart: 8, pctEnd: 25, duration: 10000 },
      { step: 2, title: `Scanning ${modeLabel} Platforms`, sub: 'Searching Internshala, Unstop, LinkedIn, RemoteOK & more', pctStart: 25, pctEnd: 50, duration: 15000 },
      { step: 3, title: 'AI Scoring & Matching', sub: `Calculating personalized match percentages for each ${modeLabel2}`, pctStart: 50, pctEnd: 92, duration: 30000 },
    ];

    const funFacts = [
      '💡 We scan 5+ job platforms including Internshala, Unstop & LinkedIn',
      `🎯 Each ${modeLabel2} is scored against your specific resume skills`,
      '🇮🇳 Results are filtered to show India-based & remote opportunities',
      '⚡ All job platforms are searched simultaneously for speed',
      '✅ Apply links go directly to the company\'s career page',
      '🧠 AI analyzes both your skills AND experience level',
      `🔥 Urgent listings are flagged so you never miss a deadline`,
      `📊 Match percentages are based on skill overlap analysis`,
    ];

    let factIndex = 0;

    // Rotate fun facts
    const factInterval = setInterval(() => {
      factIndex = (factIndex + 1) % funFacts.length;
      if (factEl) {
        factEl.style.opacity = '0';
        setTimeout(() => {
          factEl.textContent = funFacts[factIndex];
          factEl.style.opacity = '1';
        }, 300);
      }
    }, 3000);

    // Animate through phases
    let elapsed = 0;
    phases.forEach((phase, i) => {
      setTimeout(() => {
        // Update step indicators
        steps.forEach((s, j) => {
          if (!s) return;
          s.classList.remove('active');
          if (j < i) s.classList.add('done');
        });
        if (steps[i]) steps[i].classList.add('active');

        // Update title
        if (titleEl) titleEl.textContent = phase.title;
        if (subEl) subEl.textContent = phase.sub;

        // Animate percentage smoothly
        animatePct(pctEl, fillEl, phase.pctStart, phase.pctEnd, phase.duration);
      }, elapsed);

      elapsed += phase.duration;
    });

    // After all phases, hold at 94% (page will redirect)
    setTimeout(() => {
      steps.forEach(s => { if (s) s.classList.add('done'); });
      if (titleEl) titleEl.textContent = 'Preparing Your Dashboard';
      if (subEl) subEl.textContent = 'Almost there...';
      clearInterval(factInterval);
    }, elapsed);
  }

  function animatePct(pctEl, fillEl, from, to, duration) {
    const start = performance.now();
    function update(now) {
      const progress = Math.min((now - start) / duration, 1);
      // Ease out cubic for natural feel
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = Math.round(from + (to - from) * eased);
      if (pctEl) pctEl.textContent = current + '%';
      if (fillEl) fillEl.style.width = current + '%';
      if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
  }

});
