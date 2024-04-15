// Send a message to the background script to detect datasets in a large dataset
function detectDatasets() {
    chrome.runtime.sendMessage({ action: 'detect' }, (response) => {
      console.log(response);
    });
  }
  
  // Add a click event listener to the detect button
  document.getElementById('detect-button').addEventListener('click', detectDatasets);
