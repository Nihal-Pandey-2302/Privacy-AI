// Extract the large dataset from the current tab
function extractDataset() {
    const dataset = document.querySelector('pre').innerText;
    return dataset;
  }
  
  // Send the large dataset to the background script
  function sendDatasetToBackground(dataset) {
    chrome.runtime.sendMessage({ action: 'extract', dataset: dataset }, (response) => {
      console.log(response);
    });
  }
  
  // Detect datasets in the large dataset
  function detectDatasets() {
    const dataset = extractDataset();
    sendDatasetToBackground(dataset);
  }
  
  // Listen for messages from the background script
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'extract') {
      detectDatasets();
    }
  });
