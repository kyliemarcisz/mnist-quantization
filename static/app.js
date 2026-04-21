// app.js
// Handles drawing on the canvas and sending it to Flask

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, 280, 280);  // white background
ctx.strokeStyle = 'black';
ctx.lineWidth = 18;
ctx.lineCap = 'round';

let drawing = false;

// Start drawing when mouse is pressed
canvas.addEventListener('mousedown', (e) => {
    drawing = true;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
});

// Draw as mouse moves
canvas.addEventListener('mousemove', (e) => {
    if (!drawing) return;
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
});

// Stop drawing when mouse released
canvas.addEventListener('mouseup', () => drawing = false);

function clearCanvas() {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, 280, 280);
    document.getElementById('results').innerHTML = '';
}

async function predict() {
    // Shrink the 280x280 canvas down to 8x8 pixels (what our model expects)
    const small = document.createElement('canvas');
    small.width = 8; small.height = 8;
    small.getContext('2d').drawImage(canvas, 0, 0, 8, 8);

    // Extract pixel brightness values
    const imageData = small.getContext('2d').getImageData(0, 0, 8, 8).data;
    const pixels = [];
    for (let i = 0; i < imageData.length; i += 4) {
        pixels.push(255 - imageData[i]);  // invert: black drawing on white bg
    }

    // Send to Flask and get predictions back
    const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pixels })
    });

    const results = await response.json();

    // Display results on the page
    let html = '';
    for (const [model, data] of Object.entries(results)) {
        html += `<div class="result">
            ${model}: predicted <span>${data.prediction}</span>
            (${data.confidence}% confident)
        </div>`;
    }
    document.getElementById('results').innerHTML = html;
}