document.getElementById("analyze").addEventListener("click", async () => {
  const status = document.getElementById("status");
  status.textContent = "Extracting product details...";

  const [tab] = await chrome.tabs.query({
    active: true,
    currentWindow: true
  });

  const results = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: extractProductData
  });

  const productData = results[0].result;

  status.textContent = "Analyzing product...";

  try {
    const response = await fetch("http://localhost:8000/api/v1/analyze", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        product_data: productData,
        question: "Should I buy this product?"
      })
    });

    const data = await response.json();
    status.textContent = data.answer;
  } catch (error) {
    status.textContent = "Backend is not running.";
  }
});

function extractProductData() {
  return {
    title: document.querySelector("#productTitle")?.innerText?.trim() || "Not found",
    price: document.querySelector(".a-price .a-offscreen")?.innerText?.trim() || "Not found",
    rating: document.querySelector(".a-icon-alt")?.innerText?.trim() || "Not found",
    description: document.querySelector("#productDescription")?.innerText?.trim() || "Not found",
    url: window.location.href
  };
}