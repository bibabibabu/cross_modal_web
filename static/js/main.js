function bindPreview(inputId, previewId) {
  const input = document.getElementById(inputId);
  const preview = document.getElementById(previewId);
  if (!input || !preview) return;

  input.addEventListener('change', (event) => {
    const file = event.target.files && event.target.files[0];
    if (!file) {
      preview.style.display = 'none';
      preview.removeAttribute('src');
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    preview.src = objectUrl;
    preview.style.display = 'block';
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
    submitBtn.textContent = 'Detecting...';
  });
}

bindPreview('rgb_image', 'rgb-preview');
bindPreview('ir_image', 'ir-preview');
bindLoadingState();
