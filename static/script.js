function analyze() {
    // Get the input text from the textarea
    const text = document.getElementById('inputText').value;
    
    // Make a POST request to the Flask backend
    fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        // Display the sentiment analysis result
        document.getElementById('result').innerText = `Sentiment: ${data.sentiment}`;
    })
    .catch(error => {
        console.error('Error:', error);
    });
}
