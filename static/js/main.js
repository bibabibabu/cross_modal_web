function bindPreview(inputId, previewId, nameId) {
  const input = document.getElementById(inputId);
  const preview = document.getElementById(previewId);
  const name = document.getElementById(nameId);
  if (!input || !preview) return;

  input.addEventListener('change', (event) => {
    const file = event.target.files && event.target.files[0];
    if (!file) {
      preview.style.display = 'none';
      preview.removeAttribute('src');
      if (name) name.textContent = '未选择文件';
      return;
    }

    preview.src = URL.createObjectURL(file);
    preview.style.display = 'block';
    if (name) name.textContent = file.name;
  });
}

function bindLoadingState() {
  const form = document.getElementById('detect-form');
  const indicator = document.getElementById('loading-indicator');
  const submitBtn = document.getElementById('submit-btn');
  if (!form || !indicator || !submitBtn) return;

  form.addEventListener('submit', () => {
    indicator.style.display = 'flex';
    submitBtn.disabled = true;
    submitBtn.textContent = '正在检测，请稍候...';
  });
}

function bindManualPriority(selectId, inputId) {
  const select = document.getElementById(selectId);
  const input = document.getElementById(inputId);
  if (!select || !input) return;

  input.addEventListener('input', () => {
    select.disabled = input.value.trim().length > 0;
  });
}

bindPreview('rgb_image', 'rgb-preview', 'rgb-file-name');
bindPreview('ir_image', 'ir-preview', 'ir-file-name');
bindManualPriority('single_model_select', 'single_model_manual');
bindManualPriority('fusion_model_select', 'fusion_model_manual');
bindLoadingState();
