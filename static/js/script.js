// Dark/Light Toggle
document.getElementById('theme-toggle').addEventListener('click', () => {
  document.body.classList.toggle('dark-theme');
  document.body.classList.toggle('light-theme');
  const icon = document.getElementById('theme-toggle');
  icon.textContent = document.body.classList.contains('dark-theme') ? '☀️' : '🌙';
});

// Show popup message (use this function from any page)
function showPopup(message) {
  const popup = document.getElementById('popup');
  popup.textContent = message;
  popup.style.display = 'block';
  setTimeout(() => {
    popup.style.display = 'none';
  }, 3000);
}

// Example usage: Uncomment this line to test
// showPopup("Welcome back!");
