// --- BioTech Career AI Pro ---
// --- Main JavaScript File ---

document.addEventListener('DOMContentLoaded', () => {

    // --- 1. Theme Management (Dark/Light Mode) ---
    const themeToggle = document.getElementById('theme-toggle');
    const body = document.body;

    // Function to apply the stored theme on load
    const applyStoredTheme = () => {
        const storedTheme = localStorage.getItem('theme');
        if (storedTheme === 'dark') {
            body.classList.add('dark-mode');
            themeToggle.textContent = 'Toggle Light Mode';
        } else {
            body.classList.remove('dark-mode');
            themeToggle.textContent = 'Toggle Dark Mode';
        }
    };

    // Event listener for the theme toggle button
    themeToggle.addEventListener('click', () => {
        body.classList.toggle('dark-mode');
        // Save the new theme preference to localStorage
        if (body.classList.contains('dark-mode')) {
            localStorage.setItem('theme', 'dark');
            themeToggle.textContent = 'Toggle Light Mode';
        } else {
            localStorage.setItem('theme', 'light');
            themeToggle.textContent = 'Toggle Dark Mode';
        }
        // Update chart colors on theme change
        updateChartTheme();
    });

    // --- 2. Dynamic Data Simulation ---
    const dashboardData = {
        githubStatus: 'Active',
        githubSuccessRate: '99.1%',
        n8nStatus: 'Operational',
        n8nEndpoint: '/webhook/balaji-automation',
        n8nData: '1.2GB',
        aiPosts: 284,
        aiModels: 'GPT-4, Gemini, Claude',
        lcpValue: '1.8s',
        inpValue: '145ms',
        clsValue: '0.05',
        profileViews: '+23%',
        jobApps: '+31%',
        contentEngagement: '+42%',
        timeSaved: 18.7,
        dunsProgress: 85,
    };

    // Function to populate the dashboard with data
    const populateDashboard = () => {
        // Update text content
        document.getElementById('github-status').textContent = dashboardData.githubStatus;
        document.getElementById('github-success-rate').textContent = dashboardData.githubSuccessRate;
        document.getElementById('n8n-status').textContent = dashboardData.n8nStatus;
        document.getElementById('n8n-endpoint').textContent = dashboardData.n8nEndpoint;
        document.getElementById('n8n-data').textContent = dashboardData.n8nData;
        document.getElementById('ai-posts').textContent = dashboardData.aiPosts;
        document.getElementById('ai-models').textContent = dashboardData.aiModels;
        document.getElementById('lcp-value').textContent = dashboardData.lcpValue;
        document.getElementById('inp-value').textContent = dashboardData.inpValue;
        document.getElementById('cls-value').textContent = dashboardData.clsValue;
        document.getElementById('profile-views').textContent = dashboardData.profileViews;
        document.getElementById('job-apps').textContent = dashboardData.jobApps;
        document.getElementById('content-engagement').textContent = dashboardData.contentEngagement;
        document.getElementById('time-saved').textContent = dashboardData.timeSaved;
        document.getElementById('duns-progress').textContent = `${dashboardData.dunsProgress}%`;

        // Update progress bar
        document.getElementById('duns-progress-bar').style.width = `${dashboardData.dunsProgress}%`;

        // Update last run time every second
        setInterval(() => {
            const now = new Date();
            document.getElementById('github-last-run').textContent = now.toISOString().replace('T', ' ').substring(0, 19);
        }, 1000);
    };

    // --- 3. Interactive Chart (Chart.js) ---
    const ctx = document.getElementById('activity-chart').getContext('2d');
    let activityChart;

    const chartData = {
        labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6'],
        datasets: [{
            label: 'Posts Generated',
            data: [12, 19, 15, 25, 22, 30],
            borderColor: 'rgba(58, 123, 213, 1)',
            backgroundColor: 'rgba(58, 123, 213, 0.2)',
            borderWidth: 2,
            tension: 0.4,
            fill: true,
        }, {
            label: 'Applications Submitted',
            data: [5, 8, 6, 12, 10, 16],
            borderColor: 'rgba(58, 157, 93, 1)',
            backgroundColor: 'rgba(58, 157, 93, 0.2)',
            borderWidth: 2,
            tension: 0.4,
            fill: true,
        }]
    };

    const getChartOptions = (isDarkMode) => {
        const gridColor = isDarkMode ? 'rgba(255, 255, 255, 0.2)' : 'rgba(0, 0, 0, 0.1)';
        const textColor = isDarkMode ? '#e2e8f0' : '#1a202c';
        return {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: gridColor },
                    ticks: { color: textColor }
                },
                x: {
                    grid: { color: gridColor },
                    ticks: { color: textColor }
                }
            },
            plugins: {
                legend: {
                    labels: { color: textColor }
                }
            }
        };
    };

    const createChart = () => {
        const isDarkMode = body.classList.contains('dark-mode');
        activityChart = new Chart(ctx, {
            type: 'line',
            data: chartData,
            options: getChartOptions(isDarkMode)
        });
    };

    const updateChartTheme = () => {
        if (activityChart) {
            const isDarkMode = body.classList.contains('dark-mode');
            activityChart.options = getChartOptions(isDarkMode);
            activityChart.update();
        }
    };


    // --- 4. Initialization ---
    applyStoredTheme();
    populateDashboard();
    createChart();

    console.log('BioTech Career AI Pro dashboard initialized.');
});
