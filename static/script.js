document.getElementById('commentForm').addEventListener('submit', async function (event) {
    event.preventDefault();
    const comment = document.getElementById('comment').value;
    const progressBarContainer = document.getElementById('progress-bar-container');
    const progressBar = document.getElementById('progress-bar');
    const resultDiv = document.getElementById('result');

    // Reset UI
    resultDiv.textContent = '';
    progressBarContainer.style.display = 'block';
    progressBar.style.width = '0%';

    // Simulate progress bar
    let progress = 0;
    const progressInterval = setInterval(() => {
        if (progress >= 100) {
            clearInterval(progressInterval);
        } else {
            progress += 10;
            progressBar.style.width = `${progress}%`;
        }
    }, 100);

    // Send the comment to the backend
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ comment }),
        });
        const data = await response.json();

        // Display the result
        clearInterval(progressInterval);
        progressBar.style.width = '100%';
        resultDiv.textContent = `Prediction: ${data.class_name}`;
    } catch (error) {
        clearInterval(progressInterval);
        resultDiv.textContent = 'Error: Unable to process the comment.';
    }

    // Hide progress bar after a delay
    setTimeout(() => {
        progressBarContainer.style.display = 'none';
    }, 2000);
});
