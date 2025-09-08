document.getElementById("uploadForm").addEventListener("submit", async (e) => {
  e.preventDefault();

  const fileInput = document.getElementById("fileInput");
  const file = fileInput.files[0];

  if (!file) {
    alert("Please select a CSV file.");
    return;
  }

  console.log("Selected file:", file); // ✅ Debug check

  const formData = new FormData();
  formData.append("file", file);

  try {
    // ✅ Correct backend endpoint
    const response = await fetch("http://127.0.0.1:8000/analyze-csv/", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) throw new Error("Failed to upload file");

    const data = await response.json();
    const resultsTable = document.querySelector("#resultsTable tbody");
    resultsTable.innerHTML = "";

    data.results.forEach((item) => {
      const row = document.createElement("tr");

      // Feedback
      const feedbackCell = document.createElement("td");
      feedbackCell.textContent = item.feedback;
      row.appendChild(feedbackCell);

      // Score
      const scoreCell = document.createElement("td");
      scoreCell.textContent = item.score;
      row.appendChild(scoreCell);

      // Sentiment
      const sentimentCell = document.createElement("td");
      let sentimentClass = "";
      if (item.sentiment.toLowerCase().includes("positive")) {
        sentimentClass = "sentiment-positive";
      } else if (item.sentiment.toLowerCase().includes("negative")) {
        sentimentClass = "sentiment-negative";
      } else {
        sentimentClass = "sentiment-neutral";
      }
      sentimentCell.innerHTML = `<span class="${sentimentClass}">${item.sentiment}</span>`;
      row.appendChild(sentimentCell);

      // Emotion
      const emotionCell = document.createElement("td");
      const emojiMap = {
        joy: "😊",
        anger: "😡",
        sadness: "😢",
        optimism: "🌟",
        love: "❤️",
        surprise: "😲",
        fear: "😨",
      };
      const emotion = item.emotion ? item.emotion.toLowerCase() : "unknown";
      const emoji = emojiMap[emotion] || "🤔";
      emotionCell.innerHTML = `<span class="emotion-badge">${emoji} ${item.emotion}</span>`;
      row.appendChild(emotionCell);

      resultsTable.appendChild(row);
    });
  } catch (error) {
    alert("Error: " + error.message);
    console.error(error);
  }
});
