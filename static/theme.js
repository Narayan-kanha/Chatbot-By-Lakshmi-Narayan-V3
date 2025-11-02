// A simple theme switcher for light/dark mode

document.addEventListener('DOMContentLoaded', () => {
    const themeToggleButton = document.getElementById('theme-toggle'); // Requires a button with this ID
    
    const applyTheme = (theme) => {
        if (theme === 'light') {
            document.body.dataset.theme = 'light';
        } else {
            document.body.removeAttribute('data-theme');
        }
    };
    
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    let currentTheme = savedTheme ? savedTheme : (prefersDark ? 'dark' : 'light');

    applyTheme(currentTheme);

    if (themeToggleButton) {
        themeToggleButton.addEventListener('click', () => {
            let newTheme = document.body.dataset.theme === 'light' ? 'dark' : 'light';
            applyTheme(newTheme);
            localStorage.setItem('theme', newTheme);
        });
    }
});