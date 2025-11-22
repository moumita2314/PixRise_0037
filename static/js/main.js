document.addEventListener('DOMContentLoaded', () => {
  const uploadBtn = document.getElementById('uploadBtn');
  const actions = document.querySelectorAll('.action-btn');
  const resultsDiv = document.getElementById('results');
  let uploadedImage = '';

  uploadBtn.addEventListener('click', async () => {
    const fileInput = document.getElementById('imageInput');
    if (!fileInput.files.length) return alert('Please select an image.');

    const formData = new FormData();
    formData.append('image', fileInput.files[0]);

    const res = await fetch('/upload', { method: 'POST', body: formData });
    const data = await res.json();
    uploadedImage = data.filepath;
    document.getElementById('uploadedFile').textContent = `Uploaded: ${uploadedImage}`;
  });

  actions.forEach(btn => {
    btn.addEventListener('click', async () => {
      const action = btn.dataset.action;
      if (!uploadedImage) return alert('Upload an image first.');

      if (action === 'summarize' || action === 'caption') {
        const res = await fetch(`/${action}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: uploadedImage })
        });
        const data = await res.json();
        resultsDiv.textContent = JSON.stringify(data, null, 2);
      }

      if (action === 'super_res') {
        const scale = document.getElementById('srScale').value;
        const formData = new FormData();
        const fileInput = document.getElementById('imageInput');
        formData.append('image', fileInput.files[0]);
        formData.append('scale', scale);

        const res = await fetch('/super_res', { method: 'POST', body: formData });
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `super_res_${scale}x.png`;
        document.body.appendChild(a);
        a.click();
        a.remove();
      }
    });
  });
});
