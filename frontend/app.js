/**
 * WellnessAI - Frontend Application
 * AI-powered health and meal planning assistant with ML predictions
 * Version 2.0 - Charts & Predictions
 */

const API_BASE = 'http://localhost:8000/api';

// State
let currentUserId = 'USER_00001';
let isLoading = false;
let healthRisks = [];
let charts = {};

// DOM Elements
const elements = {
    // Sidebar - simplified stat cards
    userSelect: document.getElementById('userSelect'),
    recoveryScore: document.getElementById('recoveryScore'),
    sleepHours: document.getElementById('sleepHours'),
    hrvValue: document.getElementById('hrvValue'),
    strainValue: document.getElementById('strainValue'),
    weatherTemp: document.getElementById('weatherTemp'),
    weatherCondition: document.getElementById('weatherCondition'),
    weatherImpacts: document.getElementById('weatherImpacts'),
    locationInput: document.getElementById('locationInput'),
    updateWeather: document.getElementById('updateWeather'),
    detectLocation: document.getElementById('detectLocation'),
    showAlertsBtn: document.getElementById('showAlertsBtn'),
    alertCount: document.getElementById('alertCount'),
    
    // Views
    chatViewBtn: document.getElementById('chatViewBtn'),
    dashboardViewBtn: document.getElementById('dashboardViewBtn'),
    systemViewBtn: document.getElementById('systemViewBtn'),
    chatView: document.getElementById('chatView'),
    dashboardView: document.getElementById('dashboardView'),
    systemView: document.getElementById('systemView'),
    
    // System Status
    refreshSystemStatus: document.getElementById('refreshSystemStatus'),
    webSearchInput: document.getElementById('webSearchInput'),
    searchType: document.getElementById('searchType'),
    webSearchBtn: document.getElementById('webSearchBtn'),
    searchResults: document.getElementById('searchResults'),
    
    // Chat
    chatMessages: document.getElementById('chatMessages'),
    chatContainer: document.getElementById('chatContainer'),
    chatInput: document.getElementById('chatInput'),
    sendMessage: document.getElementById('sendMessage'),
    clearChat: document.getElementById('clearChat'),
    scrollBottomBtn: document.getElementById('scrollBottomBtn'),
    
    // Dashboard
    chartPeriod: document.getElementById('chartPeriod'),
    predictedRecovery: document.getElementById('predictedRecovery'),
    recoveryRange: document.getElementById('recoveryRange'),
    recoveryRecommendation: document.getElementById('recoveryRecommendation'),
    predictedStrain: document.getElementById('predictedStrain'),
    strainRange: document.getElementById('strainRange'),
    strainRecommendation: document.getElementById('strainRecommendation'),
    workoutSuggestions: document.getElementById('workoutSuggestions'),
    riskScore: document.getElementById('riskScore'),
    riskLevel: document.getElementById('riskLevel'),
    riskFactors: document.getElementById('riskFactors'),
    insightsGrid: document.getElementById('insightsGrid'),
    
    // Panels
    loadingOverlay: document.getElementById('loadingOverlay'),
    risksPanel: document.getElementById('risksPanel'),
    risksContent: document.getElementById('risksContent'),
    closeRisks: document.getElementById('closeRisks')
};

// Chart.js configuration
Chart.defaults.color = '#6b8f7a';
Chart.defaults.borderColor = 'rgba(74, 222, 128, 0.1)';
Chart.defaults.font.family = "'DM Sans', sans-serif";

// ============ Human-Friendly Interpretation Functions ============

function getRecoveryInterpretation(score) {
    if (score >= 80) return { text: "Ready to conquer the day!", emoji: "💪", class: "excellent" };
    if (score >= 60) return { text: "Good shape, pace yourself", emoji: "👍", class: "good" };
    if (score >= 40) return { text: "Take it easy today", emoji: "🧘", class: "moderate" };
    return { text: "Rest day recommended", emoji: "😴", class: "low" };
}

function getSleepInterpretation(hours) {
    if (hours >= 8) return { text: "Well rested!", emoji: "🌟", class: "excellent", moon: "🌕" };
    if (hours >= 7) return { text: "Good night's sleep", emoji: "😊", class: "good", moon: "🌔" };
    if (hours >= 6) return { text: "Could use more rest", emoji: "😐", class: "moderate", moon: "🌓" };
    return { text: "Sleep more tonight", emoji: "😴", class: "low", moon: "🌑" };
}

function getHrvInterpretation(hrv, baseline = 100) {
    // Higher HRV = calmer, lower stress
    const ratio = hrv / baseline;
    if (ratio >= 1.1) return { text: "Very calm & relaxed", emoji: "😌", class: "excellent" };
    if (ratio >= 0.9) return { text: "Feeling balanced", emoji: "🙂", class: "good" };
    if (ratio >= 0.7) return { text: "A bit stressed", emoji: "😟", class: "moderate" };
    return { text: "High stress - relax", emoji: "😰", class: "low" };
}

function getStrainInterpretation(strain) {
    if (strain >= 15) return { text: "Very active day!", emoji: "🏃", class: "excellent", percent: 90 };
    if (strain >= 10) return { text: "Good activity level", emoji: "🚶", class: "good", percent: 65 };
    if (strain >= 5) return { text: "Light activity", emoji: "🧘", class: "moderate", percent: 40 };
    return { text: "Move more today", emoji: "💤", class: "low", percent: 15 };
}

function getTrendText(trend) {
    const trendMap = {
        'improving': 'Getting better',
        'declining': 'Needs attention',
        'stable': 'Staying steady'
    };
    return trendMap[trend] || 'Steady';
}

// Generate Daily Summary based on all health metrics
function generateDailySummary(metrics) {
    const recovery = metrics.avg_recovery_score || 65;
    const sleep = metrics.avg_sleep_hours || 7;
    const strain = metrics.avg_strain || 10;
    
    // Determine overall state
    let overallScore = 0;
    if (recovery >= 70) overallScore += 2;
    else if (recovery >= 50) overallScore += 1;
    
    if (sleep >= 7) overallScore += 2;
    else if (sleep >= 6) overallScore += 1;
    
    // Generate summary based on score
    if (overallScore >= 4) {
        return {
            emoji: "🌟",
            headline: "You're feeling fantastic today!",
            detail: "Your body is well-rested and full of energy. Great time for a challenging workout or tackling that big project.",
            class: "excellent",
            action: "Plan a great workout"
        };
    } else if (overallScore >= 3) {
        return {
            emoji: "😊",
            headline: "You're doing well today!",
            detail: "You've got good energy levels. A moderate workout would be perfect, and remember to stay hydrated.",
            class: "good",
            action: "See today's plan"
        };
    } else if (overallScore >= 2) {
        return {
            emoji: "😐",
            headline: "Take it easy today",
            detail: "Your body could use a bit more rest. Consider light activity like walking or yoga, and prioritize sleep tonight.",
            class: "moderate",
            action: "Get recovery tips"
        };
    } else {
        return {
            emoji: "🛋️",
            headline: "Rest day recommended",
            detail: "Your body is asking for recovery. Skip intense workouts, eat nutritious foods, and aim for an early bedtime.",
            class: "low",
            action: "See what to eat"
        };
    }
}

function updateDailySummary(metrics) {
    const summary = generateDailySummary(metrics);
    
    const card = document.getElementById('dailySummaryCard');
    const emoji = document.getElementById('summaryEmoji');
    const headline = document.getElementById('summaryHeadline');
    const detail = document.getElementById('summaryDetail');
    const primaryBtn = document.getElementById('summaryPrimaryAction');
    
    if (card) card.className = `daily-summary-card ${summary.class}`;
    if (emoji) emoji.textContent = summary.emoji;
    if (headline) headline.textContent = summary.headline;
    if (detail) detail.textContent = summary.detail;
    if (primaryBtn) primaryBtn.textContent = summary.action;
}

function askPrimaryQuestion() {
    const summary = document.getElementById('summaryHeadline')?.textContent || '';
    let prompt = "Create a personalized plan for today based on my health metrics";
    
    if (summary.includes('Rest')) {
        prompt = "I need to rest today. What should I eat and do to recover?";
    } else if (summary.includes('easy')) {
        prompt = "What light activities and foods would help me recover today?";
    } else if (summary.includes('fantastic')) {
        prompt = "I'm feeling great! Give me an ambitious workout and meal plan for today.";
    }
    
    elements.chatInput.value = prompt;
    elements.chatInput.focus();
    sendChatMessage();
}

function scrollToDetails() {
    const predictions = document.querySelector('.predictions-section');
    if (predictions) {
        predictions.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// ============ Onboarding Tour ============
const onboardingSteps = [
    {
        target: '.stat-card.recovery',
        title: 'Your Body Battery',
        text: "This shows how recharged you feel - like a phone battery. The fuller it is, the more energy you have for the day!",
        position: 'right'
    },
    {
        target: '.stat-card.hrv',
        title: 'Stress Level',
        text: "We track how stressed or relaxed your body feels. The calmer emoji means you're doing great!",
        position: 'right'
    },
    {
        target: '.quick-actions',
        title: 'Quick Help',
        text: "Tap any of these to get instant advice about meals, headaches, energy, or sleep.",
        position: 'right'
    },
    {
        target: '.view-toggle',
        title: 'Switch Views',
        text: "Toggle between chatting with your health assistant and viewing your health dashboard.",
        position: 'bottom'
    },
    {
        target: '.chat-input-wrapper',
        title: 'Ask Anything',
        text: "Type any health question here - from meal planning to workout advice. I'm here to help!",
        position: 'top'
    }
];

let currentOnboardingStep = 0;

function showOnboarding() {
    const hasSeenOnboarding = localStorage.getItem('wellnessai-onboarding-complete');
    if (hasSeenOnboarding) return;
    
    // Small delay to let the page load first
    setTimeout(() => {
        const overlay = document.getElementById('onboardingOverlay');
        if (overlay) {
            overlay.style.display = 'block';
            showOnboardingStep(0);
        }
    }, 1500);
}

function showOnboardingStep(stepIndex) {
    if (stepIndex >= onboardingSteps.length) {
        completeOnboarding();
        return;
    }
    
    currentOnboardingStep = stepIndex;
    const step = onboardingSteps[stepIndex];
    const targetElement = document.querySelector(step.target);
    
    if (!targetElement) {
        showOnboardingStep(stepIndex + 1);
        return;
    }
    
    // Remove previous highlights
    document.querySelectorAll('.onboarding-highlight').forEach(el => {
        el.classList.remove('onboarding-highlight');
    });
    
    // Highlight current element
    targetElement.classList.add('onboarding-highlight');
    
    // Update tooltip content
    document.getElementById('onboardingStepIndicator').textContent = `Step ${stepIndex + 1} of ${onboardingSteps.length}`;
    document.getElementById('onboardingTitle').textContent = step.title;
    document.getElementById('onboardingText').textContent = step.text;
    
    const nextBtn = document.getElementById('onboardingNext');
    nextBtn.textContent = stepIndex === onboardingSteps.length - 1 ? "Get Started" : "Next";
    
    // Position tooltip near the target
    positionOnboardingTooltip(targetElement, step.position);
}

function positionOnboardingTooltip(targetElement, position) {
    const tooltip = document.getElementById('onboardingTooltip');
    const rect = targetElement.getBoundingClientRect();
    
    const tooltipWidth = 320;
    const tooltipHeight = tooltip.offsetHeight || 200;
    const margin = 20;
    
    let top, left;
    
    switch (position) {
        case 'right':
            top = rect.top + (rect.height / 2) - (tooltipHeight / 2);
            left = rect.right + margin;
            break;
        case 'left':
            top = rect.top + (rect.height / 2) - (tooltipHeight / 2);
            left = rect.left - tooltipWidth - margin;
            break;
        case 'top':
            top = rect.top - tooltipHeight - margin;
            left = rect.left + (rect.width / 2) - (tooltipWidth / 2);
            break;
        case 'bottom':
        default:
            top = rect.bottom + margin;
            left = rect.left + (rect.width / 2) - (tooltipWidth / 2);
            break;
    }
    
    // Keep tooltip within viewport
    top = Math.max(20, Math.min(top, window.innerHeight - tooltipHeight - 20));
    left = Math.max(20, Math.min(left, window.innerWidth - tooltipWidth - 20));
    
    tooltip.style.top = `${top}px`;
    tooltip.style.left = `${left}px`;
}

function nextOnboardingStep() {
    showOnboardingStep(currentOnboardingStep + 1);
}

function completeOnboarding() {
    localStorage.setItem('wellnessai-onboarding-complete', 'true');
    const overlay = document.getElementById('onboardingOverlay');
    if (overlay) overlay.style.display = 'none';
    document.querySelectorAll('.onboarding-highlight').forEach(el => {
        el.classList.remove('onboarding-highlight');
    });
    showNotification('Welcome! Ask me anything about your health.', 'success');
}

function setupOnboarding() {
    const nextBtn = document.getElementById('onboardingNext');
    const skipBtn = document.getElementById('onboardingSkip');
    
    if (nextBtn) nextBtn.addEventListener('click', nextOnboardingStep);
    if (skipBtn) skipBtn.addEventListener('click', completeOnboarding);
}

// ============ Floating Ask Button ============
let currentView = 'chat';

function setupFloatingButton() {
    const floatingBtn = document.getElementById('floatingAskBtn');
    if (floatingBtn) {
        floatingBtn.addEventListener('click', () => {
            if (currentView !== 'chat') {
                // Generate a contextual question based on current health data
                const contextualQuestion = generateContextualQuestion();
                switchView('chat');
                
                // Pre-fill the chat input with contextual question
                if (contextualQuestion) {
                    elements.chatInput.value = contextualQuestion;
                    elements.chatInput.focus();
                    // Auto-resize the input
                    elements.chatInput.style.height = 'auto';
                    elements.chatInput.style.height = elements.chatInput.scrollHeight + 'px';
                }
            }
        });
        
        // Initially hide on chat view
        updateFloatingButtonVisibility();
    }
}

function generateContextualQuestion() {
    // Generate a relevant question based on current health metrics
    const recovery = elements.recoveryScore?.textContent;
    const sleep = elements.sleepHours?.textContent;
    
    const questions = [];
    
    if (recovery && parseFloat(recovery) < 60) {
        questions.push("My recovery is low today. What should I eat to help me recover?");
        questions.push("I'm not feeling my best. What activities would you recommend?");
    } else if (recovery && parseFloat(recovery) >= 80) {
        questions.push("I'm feeling great today! What's a good challenging workout?");
        questions.push("My energy is high. What should I eat to maintain this?");
    }
    
    if (sleep && parseFloat(sleep) < 6) {
        questions.push("I didn't sleep well. What foods help with energy?");
        questions.push("How can I improve my sleep quality tonight?");
    }
    
    // Add some general questions as fallback
    questions.push("What should I eat today based on my health data?");
    questions.push("Give me a personalized meal plan for today");
    questions.push("What are my health insights for this week?");
    
    // Return a random question from the relevant ones
    return questions[Math.floor(Math.random() * Math.min(questions.length, 3))];
}

function updateFloatingButtonVisibility() {
    const floatingBtn = document.getElementById('floatingAskBtn');
    if (floatingBtn) {
        if (currentView === 'chat') {
            floatingBtn.style.display = 'none';
        } else {
            floatingBtn.style.display = 'flex';
        }
    }
}

// ============ Celebrations ============
let previousMetrics = {};

function checkForImprovements(newMetrics) {
    if (!previousMetrics.avg_recovery_score) {
        previousMetrics = { ...newMetrics };
        return;
    }
    
    // Check recovery improvement
    if (newMetrics.avg_recovery_score && previousMetrics.avg_recovery_score) {
        const improvement = newMetrics.avg_recovery_score - previousMetrics.avg_recovery_score;
        if (improvement >= 5) {
            showCelebration('energy levels', Math.round(improvement));
        }
    }
    
    // Check sleep improvement
    if (newMetrics.avg_sleep_hours && previousMetrics.avg_sleep_hours) {
        const improvement = newMetrics.avg_sleep_hours - previousMetrics.avg_sleep_hours;
        if (improvement >= 0.5) {
            showCelebration('sleep', `${improvement.toFixed(1)} hours`);
        }
    }
    
    previousMetrics = { ...newMetrics };
}

function showCelebration(metric, improvement) {
    const message = typeof improvement === 'number' 
        ? `Great job! Your ${metric} improved by ${improvement}%`
        : `Great job! Your ${metric} improved by ${improvement}`;
    showNotification(message, 'celebration');
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    init();
    setupOnboarding();
    setupFloatingButton();
    showOnboarding();
});

async function init() {
    showLoadingState(true);
    
    try {
        await Promise.all([
            loadUsers(),
            detectUserLocation() // Auto-detect location on startup
        ]);
        await loadUserData();
    } catch (error) {
        console.error('Initialization error:', error);
        showNotification('Failed to initialize. Please refresh the page.', 'error');
    } finally {
        showLoadingState(false);
    }
    
    setupEventListeners();
    setupScrollObserver();
    initializeCharts();
}

// Event Listeners
function setupEventListeners() {
    // View toggle
    elements.chatViewBtn.addEventListener('click', () => switchView('chat'));
    elements.dashboardViewBtn.addEventListener('click', () => switchView('dashboard'));
    if (elements.systemViewBtn) {
        elements.systemViewBtn.addEventListener('click', () => switchView('system'));
    }
    
    // System status refresh
    if (elements.refreshSystemStatus) {
        elements.refreshSystemStatus.addEventListener('click', loadSystemStatus);
    }
    
    // Web search
    if (elements.webSearchBtn) {
        elements.webSearchBtn.addEventListener('click', performWebSearch);
    }
    if (elements.webSearchInput) {
        elements.webSearchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                performWebSearch();
            }
        });
    }

    // User selection
    elements.userSelect.addEventListener('change', async (e) => {
        currentUserId = e.target.value;
        showLoadingState(true);
        await loadUserData();
        if (elements.dashboardView.classList.contains('active')) {
            await loadDashboardData();
        }
        showLoadingState(false);
        addSystemMessage(`Switched to profile ${formatUserId(currentUserId)}. Health data updated.`);
    });

    // Send message
    elements.sendMessage.addEventListener('click', sendChatMessage);
    elements.chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendChatMessage();
        }
    });

    // Auto-resize textarea
    elements.chatInput.addEventListener('input', () => {
        elements.chatInput.style.height = 'auto';
        elements.chatInput.style.height = Math.min(elements.chatInput.scrollHeight, 160) + 'px';
    });

    // Clear chat
    elements.clearChat.addEventListener('click', clearConversation);

    // Weather update
    elements.updateWeather.addEventListener('click', async () => {
        elements.updateWeather.classList.add('loading');
        await loadWeather();
        elements.updateWeather.classList.remove('loading');
    });

    // Detect location
    elements.detectLocation.addEventListener('click', async () => {
        await detectUserLocation();
    });
    
    elements.locationInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') loadWeather();
    });

    // Quick actions
    document.querySelectorAll('.action-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const prompt = btn.dataset.prompt;
            if (prompt) {
                switchView('chat');
                elements.chatInput.value = prompt;
                elements.chatInput.focus();
                sendChatMessage();
            }
        });
    });

    // Health alerts panel
    elements.showAlertsBtn.addEventListener('click', () => {
        elements.risksPanel.classList.add('open');
        elements.risksPanel.setAttribute('aria-hidden', 'false');
    });
    
    elements.closeRisks.addEventListener('click', () => {
        elements.risksPanel.classList.remove('open');
        elements.risksPanel.setAttribute('aria-hidden', 'true');
    });
    
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && elements.risksPanel.classList.contains('open')) {
            elements.risksPanel.classList.remove('open');
            elements.risksPanel.setAttribute('aria-hidden', 'true');
        }
    });

    // Scroll to bottom button
    elements.scrollBottomBtn.addEventListener('click', scrollToBottom);
    
    // Stat cards - click to ask about that metric
    document.querySelectorAll('.stat-card').forEach(card => {
        card.addEventListener('click', () => {
            const statType = card.dataset.type || card.classList[1];
            askAboutStat(statType);
        });
        
        card.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                const statType = card.dataset.type || card.classList[1];
                askAboutStat(statType);
            }
        });
    });

    // Chart period change
    elements.chartPeriod.addEventListener('change', loadDashboardData);
}

// View switching
function switchView(view) {
    // Remove active from all buttons and views
    elements.chatViewBtn.classList.remove('active');
    elements.dashboardViewBtn.classList.remove('active');
    if (elements.systemViewBtn) elements.systemViewBtn.classList.remove('active');
    
    elements.chatView.classList.remove('active');
    elements.chatView.style.display = 'none';
    elements.dashboardView.classList.remove('active');
    elements.dashboardView.style.display = 'none';
    if (elements.systemView) {
        elements.systemView.classList.remove('active');
        elements.systemView.style.display = 'none';
    }
    
    if (view === 'chat') {
        elements.chatViewBtn.classList.add('active');
        elements.chatView.classList.add('active');
        elements.chatView.style.display = 'flex';
    } else if (view === 'dashboard') {
        elements.dashboardViewBtn.classList.add('active');
        elements.dashboardView.classList.add('active');
        elements.dashboardView.style.display = 'flex';
        loadDashboardData();
    } else if (view === 'system') {
        if (elements.systemViewBtn) elements.systemViewBtn.classList.add('active');
        if (elements.systemView) {
            elements.systemView.classList.add('active');
            elements.systemView.style.display = 'flex';
            loadSystemStatus();
        }
    }
    
    // Track current view and update floating button visibility
    currentView = view;
    updateFloatingButtonVisibility();
}

// Scroll observer
function setupScrollObserver() {
    elements.chatContainer.addEventListener('scroll', () => {
        const { scrollTop, scrollHeight, clientHeight } = elements.chatContainer;
        const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;
        
        if (isNearBottom) {
            elements.scrollBottomBtn.classList.remove('visible');
        } else {
            elements.scrollBottomBtn.classList.add('visible');
        }
    });
}

// ============ System Status Functions ============
async function loadSystemStatus() {
    try {
        const response = await fetch(`${API_BASE}/system-status`);
        const data = await response.json();
        
        // Update LLM status
        const llmStatus = document.getElementById('llmStatus');
        if (llmStatus) {
            const statusEl = llmStatus.querySelector('.component-status');
            const modelEl = document.getElementById('llmModel');
            const deviceEl = document.getElementById('llmDevice');
            
            if (data.components.local_llm.status === 'active') {
                statusEl.textContent = '✓ Active';
                statusEl.className = 'component-status active';
            } else {
                statusEl.textContent = '✗ Disabled';
                statusEl.className = 'component-status error';
            }
            
            if (modelEl) modelEl.textContent = data.components.local_llm.model || 'Not loaded';
            if (deviceEl) deviceEl.textContent = data.components.local_llm.device || 'CPU';
        }
        
        // Update Recommendation Engine status
        const recStatus = document.getElementById('recEngineStatus');
        if (recStatus) {
            const statusEl = recStatus.querySelector('.component-status');
            statusEl.textContent = '✓ Active';
            statusEl.className = 'component-status active';
        }
        
        // Update ML status
        const mlStatus = document.getElementById('mlStatus');
        if (mlStatus) {
            const statusEl = mlStatus.querySelector('.component-status');
            statusEl.textContent = '✓ Active';
            statusEl.className = 'component-status active';
        }
        
        // Update Web Search status
        const webSearchStatus = document.getElementById('webSearchStatus');
        if (webSearchStatus) {
            const statusEl = webSearchStatus.querySelector('.component-status');
            statusEl.textContent = '✓ Available';
            statusEl.className = 'component-status active';
        }
        
    } catch (error) {
        console.error('Failed to load system status:', error);
    }
}

async function performWebSearch() {
    const query = elements.webSearchInput?.value?.trim();
    const searchType = elements.searchType?.value || 'general';
    
    if (!query) {
        showNotification('Please enter a search query', 'error');
        return;
    }
    
    // Show loading
    if (elements.searchResults) {
        elements.searchResults.innerHTML = '<p class="search-placeholder">🔍 Searching...</p>';
    }
    if (elements.webSearchBtn) {
        elements.webSearchBtn.disabled = true;
        elements.webSearchBtn.textContent = 'Searching...';
    }
    
    try {
        const response = await fetch(`${API_BASE}/web-search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, search_type: searchType })
        });
        
        const data = await response.json();
        
        if (data.success && data.results.length > 0) {
            let html = '';
            
            if (data.summary) {
                html += `<div class="search-result-item">
                    <div class="search-result-title">Summary</div>
                    <div class="search-result-snippet">${data.summary}</div>
                    ${data.source ? `<div class="search-result-source">Source: ${data.source}</div>` : ''}
                </div>`;
            }
            
            data.results.forEach(result => {
                html += `<div class="search-result-item">
                    <div class="search-result-title">${result.title || 'Result'}</div>
                    <div class="search-result-snippet">${result.snippet}</div>
                    ${result.url ? `<div class="search-result-source"><a href="${result.url}" target="_blank">🔗 ${result.source || 'View source'}</a></div>` : ''}
                </div>`;
            });
            
            elements.searchResults.innerHTML = html;
        } else {
            elements.searchResults.innerHTML = `<p class="search-placeholder">No results found for "${query}". Try a different search term.</p>`;
        }
        
    } catch (error) {
        console.error('Web search error:', error);
        elements.searchResults.innerHTML = '<p class="search-placeholder">❌ Search failed. Please try again.</p>';
    } finally {
        if (elements.webSearchBtn) {
            elements.webSearchBtn.disabled = false;
            elements.webSearchBtn.textContent = 'Search';
        }
    }
}

// Ask about a specific stat
function askAboutStat(statType) {
    const prompts = {
        // New simplified stat types
        energy: "My energy levels are showing as 77%. How can I boost my energy through nutrition and lifestyle?",
        recovery: "Analyze my recovery score and give me specific recommendations to improve it through nutrition and lifestyle changes.",
        sleep: "My sleep metrics need improvement. What dietary changes and habits can help me sleep better and longer?",
        stress: "What can I do to manage my stress levels? Give me nutrition and lifestyle recommendations.",
        hrv: "Tell me about my HRV trends and what foods or supplements can help improve my heart rate variability.",
        activity: "Based on my activity level, what should my nutrition strategy be today?",
        strain: "Based on my strain levels, what should my nutrition strategy be to support my activity and recovery?"
    };
    
    const prompt = prompts[statType];
    if (prompt) {
        switchView('chat');
        elements.chatInput.value = prompt;
        elements.chatInput.focus();
        sendChatMessage();
    }
}

// API Functions
async function loadUsers() {
    try {
        const response = await fetch(`${API_BASE}/users?limit=50`);
        const data = await response.json();
        
        elements.userSelect.innerHTML = data.users.map(userId => 
            `<option value="${userId}">${formatUserId(userId)}</option>`
        ).join('');
        
        currentUserId = data.users[0] || 'USER_00001';
    } catch (error) {
        console.error('Failed to load users:', error);
        elements.userSelect.innerHTML = '<option value="USER_00001">User 1 (Default)</option>';
    }
}

function formatUserId(userId) {
    const match = userId.match(/USER_(\d+)/);
    if (match) return `User ${parseInt(match[1], 10)}`;
    return userId;
}

async function loadUserData() {
    try {
        const response = await fetch(`${API_BASE}/users/${currentUserId}/health-context`);
        if (!response.ok) throw new Error('User not found');
        
        const data = await response.json();
        updateHealthStats(data);
        updateHealthRisks(data.health_risks || []);
    } catch (error) {
        console.error('Failed to load user data:', error);
        showNotification('Failed to load health data', 'error');
    }
}

async function loadWeather() {
    const location = elements.locationInput.value || 'New York';
    
    try {
        const response = await fetch(`${API_BASE}/weather?location=${encodeURIComponent(location)}`);
        const data = await response.json();
        updateWeatherDisplay(data);
    } catch (error) {
        console.error('Failed to load weather:', error);
        elements.weatherCondition.textContent = 'Unable to load';
    }
}

async function detectUserLocation() {
    if (!navigator.geolocation) {
        showNotification('Geolocation is not supported by your browser', 'error');
        elements.locationInput.value = 'New York';
        return loadWeather();
    }

    elements.detectLocation.classList.add('loading');
    elements.locationInput.placeholder = 'Detecting location...';
    elements.weatherCondition.textContent = 'Detecting...';

    return new Promise((resolve) => {
        navigator.geolocation.getCurrentPosition(
            async (position) => {
                const { latitude, longitude } = position.coords;
                // WeatherAPI.com accepts lat,lon format
                const locationQuery = `${latitude},${longitude}`;
                
                try {
                    const response = await fetch(`${API_BASE}/weather?location=${encodeURIComponent(locationQuery)}`);
                    const data = await response.json();
                    
                    // Update the input with the detected city name
                    if (data.weather_summary?.location) {
                        elements.locationInput.value = data.weather_summary.location;
                    } else {
                        elements.locationInput.value = locationQuery;
                    }
                    
                    updateWeatherDisplay(data);
                    elements.detectLocation.classList.remove('loading');
                    showNotification(`Location detected: ${elements.locationInput.value}`, 'success');
                    resolve(data);
                } catch (error) {
                    console.error('Failed to fetch weather for detected location:', error);
                    elements.locationInput.value = 'New York';
                    elements.detectLocation.classList.remove('loading');
                    await loadWeather();
                    resolve(null);
                }
            },
            (error) => {
                console.warn('Geolocation error:', error.message);
                elements.detectLocation.classList.remove('loading');
                
                let errorMessage = 'Could not detect location';
                if (error.code === error.PERMISSION_DENIED) {
                    errorMessage = 'Location access denied. Using default.';
                } else if (error.code === error.POSITION_UNAVAILABLE) {
                    errorMessage = 'Location unavailable. Using default.';
                } else if (error.code === error.TIMEOUT) {
                    errorMessage = 'Location request timed out. Using default.';
                }
                
                showNotification(errorMessage, 'warning');
                elements.locationInput.value = 'New York';
                loadWeather();
                resolve(null);
            },
            {
                enableHighAccuracy: false,
                timeout: 10000,
                maximumAge: 300000 // Cache for 5 minutes
            }
        );
    });
}

async function loadDashboardData() {
    const days = parseInt(elements.chartPeriod.value) || 30;
    
    try {
        const response = await fetch(`${API_BASE}/users/${currentUserId}/dashboard?days=${days}`);
        const data = await response.json();
        
        // Update Daily Summary Card first (most important)
        if (data.recent_metrics) {
            updateDailySummary(data.recent_metrics);
        }
        
        updatePredictions(data.predictions);
        updateCharts(data.trends);
        updateInsights(data.correlations);
    } catch (error) {
        console.error('Failed to load dashboard data:', error);
        showNotification('Failed to load dashboard data', 'error');
    }
}

async function sendChatMessage() {
    const message = elements.chatInput.value.trim();
    if (!message || isLoading) return;

    isLoading = true;
    elements.sendMessage.disabled = true;
    
    addMessage(message, 'user');
    elements.chatInput.value = '';
    elements.chatInput.style.height = 'auto';

    const typingId = addTypingIndicator();

    try {
        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                user_id: currentUserId,
                include_weather: true,
                location: elements.locationInput.value || 'New York'
            })
        });

        const data = await response.json();
        removeTypingIndicator(typingId);
        
        setTimeout(() => {
            const msgEl = addMessage(data.response, 'assistant');
            if (data.ui_components && data.ui_components.length && msgEl) {
                const contentDiv = msgEl.querySelector('.message-content');
                if (contentDiv) {
                    data.ui_components.forEach(comp => contentDiv.appendChild(renderUIComponent(comp)));
                }
            }
            if (data.health_summary) updateQuickStats(data.health_summary);
        }, 200);
        
    } catch (error) {
        console.error('Chat error:', error);
        removeTypingIndicator(typingId);
        addMessage('I apologize, but I encountered an error processing your request. Please try again.', 'assistant');
    } finally {
        isLoading = false;
        elements.sendMessage.disabled = false;
        elements.chatInput.focus();
    }
}

async function clearConversation() {
    if (!confirm('Are you sure you want to clear the chat history?')) return;
    
    try {
        await fetch(`${API_BASE}/chat/clear?user_id=${currentUserId}`, { method: 'POST' });
        
        const messages = elements.chatMessages.querySelectorAll('.message');
        messages.forEach((msg, index) => {
            if (index > 0) {
                msg.style.animation = 'messageOut 0.3s ease forwards';
                setTimeout(() => msg.remove(), 300);
            }
        });
        
        showNotification('Conversation cleared', 'success');
    } catch (error) {
        console.error('Failed to clear conversation:', error);
        showNotification('Failed to clear conversation', 'error');
    }
}

// UI Update Functions
function updateHealthStats(data) {
    const metrics = data.recent_metrics || {};
    
    // Energy (Recovery)
    if (metrics.avg_recovery_score !== undefined) {
        const recovery = Math.round(metrics.avg_recovery_score);
        elements.recoveryScore.textContent = recovery;
        
        const card = document.querySelector('[data-type="recovery"]');
        const statusEl = document.getElementById('recoveryStatus');
        const emojiEl = document.getElementById('recoveryEmoji');
        
        let status, emoji, level;
        if (recovery >= 80) {
            status = "Feeling great today";
            emoji = "⚡";
            level = "good";
        } else if (recovery >= 60) {
            status = "Good energy levels";
            emoji = "🔋";
            level = "good";
        } else if (recovery >= 40) {
            status = "Take it easy today";
            emoji = "🔋";
            level = "warning";
        } else {
            status = "Rest day recommended";
            emoji = "🪫";
            level = "alert";
        }
        
        if (statusEl) statusEl.textContent = status;
        if (emojiEl) emojiEl.textContent = emoji;
        if (card) card.setAttribute('data-status', level);
    }
    
    // Sleep
    if (metrics.avg_sleep_hours !== undefined) {
        const sleep = metrics.avg_sleep_hours;
        elements.sleepHours.textContent = sleep.toFixed(1);
        
        const card = document.querySelector('[data-type="sleep"]');
        const statusEl = document.getElementById('sleepStatus');
        const emojiEl = document.getElementById('sleepEmoji');
        
        let status, emoji, level;
        if (sleep >= 8) {
            status = "Well rested";
            emoji = "😴";
            level = "good";
        } else if (sleep >= 7) {
            status = "Good night's rest";
            emoji = "😴";
            level = "good";
        } else if (sleep >= 6) {
            status = "Could use more sleep";
            emoji = "😪";
            level = "warning";
        } else {
            status = "Sleep deficit";
            emoji = "😵";
            level = "alert";
        }
        
        if (statusEl) statusEl.textContent = status;
        if (emojiEl) emojiEl.textContent = emoji;
        if (card) card.setAttribute('data-status', level);
    }
    
    // Stress (HRV)
    if (metrics.avg_hrv !== undefined) {
        const hrv = Math.round(metrics.avg_hrv);
        elements.hrvValue.textContent = hrv;
        
        const card = document.querySelector('[data-type="stress"]');
        const statusEl = document.getElementById('stressStatus');
        const emojiEl = document.getElementById('stressEmoji');
        
        let status, emoji, level;
        if (hrv >= 60) {
            status = "Very relaxed";
            emoji = "😌";
            level = "good";
        } else if (hrv >= 45) {
            status = "Balanced";
            emoji = "🙂";
            level = "good";
        } else if (hrv >= 30) {
            status = "Some tension";
            emoji = "😐";
            level = "warning";
        } else {
            status = "High stress";
            emoji = "😰";
            level = "alert";
        }
        
        if (statusEl) statusEl.textContent = status;
        if (emojiEl) emojiEl.textContent = emoji;
        if (card) card.setAttribute('data-status', level);
    }
    
    // Activity (Strain)
    if (metrics.avg_strain !== undefined) {
        const strain = metrics.avg_strain;
        elements.strainValue.textContent = strain.toFixed(1);
        
        const card = document.querySelector('[data-type="activity"]');
        const statusEl = document.getElementById('activityStatus');
        const emojiEl = document.getElementById('activityEmoji');
        
        let status, emoji, level;
        if (strain >= 15) {
            status = "Very active";
            emoji = "🔥";
            level = "good";
        } else if (strain >= 10) {
            status = "Active day";
            emoji = "🏃";
            level = "good";
        } else if (strain >= 5) {
            status = "Light activity";
            emoji = "🚶";
            level = "warning";
        } else {
            status = "Rest day";
            emoji = "🧘";
            level = "good";
        }
        
        if (statusEl) statusEl.textContent = status;
        if (emojiEl) emojiEl.textContent = emoji;
        if (card) card.setAttribute('data-status', level);
    }
}

function updateTrend(element, trend, isCritical = false) {
    // Use friendly language instead of technical terms
    const friendlyTrend = getTrendText(trend);
    element.textContent = friendlyTrend;
    element.className = 'stat-trend';
    
    if (isCritical && trend === 'declining') {
        element.classList.add('critical');
    } else if (trend) {
        element.classList.add(trend);
    } else {
        element.classList.add('stable');
    }
}

function updateQuickStats(summary) {
    if (summary.recovery_score) elements.recoveryScore.textContent = Math.round(summary.recovery_score);
    if (summary.sleep_hours) elements.sleepHours.textContent = summary.sleep_hours.toFixed(1);
    if (summary.hrv) elements.hrvValue.textContent = Math.round(summary.hrv);
}

function updateWeatherDisplay(data) {
    const weather = data.weather_summary || data;
    
    if (weather.temperature !== undefined) {
        elements.weatherTemp.textContent = `${Math.round(weather.temperature)}°C`;
    }
    
    if (weather.condition) {
        elements.weatherCondition.textContent = weather.condition;
    }
    
    const impacts = data.health_impacts || [];
    elements.weatherImpacts.innerHTML = impacts.map(impact => 
        `<span class="weather-impact-tag" role="listitem">${getImpactIcon(impact)} ${formatImpact(impact)}</span>`
    ).join('');
}

function getImpactIcon(impact) {
    const icons = {
        'cold_stress': '❄️', 'heat_stress': '🔥', 'high_humidity': '💧',
        'low_humidity': '🏜️', 'low_light': '☁️', 'low_pressure': '🌀',
        'high_pressure': '☀️', 'cold_dry_season': '🍂'
    };
    return icons[impact] || '•';
}

function formatImpact(impact) {
    return impact.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

function updateHealthRisks(risks) {
    healthRisks = risks;
    elements.alertCount.textContent = risks.length;
    
    if (risks.length === 0) {
        elements.showAlertsBtn.style.display = 'none';
        elements.risksContent.innerHTML = `
            <div style="text-align: center; padding: 40px 20px; color: var(--text-muted);">
                <div style="font-size: 3rem; margin-bottom: 16px;">🎉</div>
                <p style="font-size: 1rem;">No health alerts detected!</p>
                <p style="font-size: 0.85rem; margin-top: 8px;">Your metrics are looking healthy.</p>
            </div>
        `;
    } else {
        elements.showAlertsBtn.style.display = 'flex';
        elements.risksContent.innerHTML = risks.map(risk => `
            <div class="risk-item ${risk.severity}" role="listitem">
                <h4>${risk.risk}</h4>
                <p>${risk.description}</p>
                <div class="risk-dietary">
                    <strong>Dietary Recommendation</strong>
                    ${risk.dietary_impact}
                </div>
            </div>
        `).join('');
    }
}

// ============ Dashboard Functions ============
function updatePredictions(predictions) {
    if (!predictions) return;
    
    // Tomorrow's Outlook (Recovery) - Friendly Version
    const recovery = predictions.recovery || {};
    const predictedScore = recovery.predicted_recovery || 65;
    
    // Update hidden technical values
    if (elements.predictedRecovery) elements.predictedRecovery.textContent = predictedScore;
    
    // Update friendly summary
    const tomorrowSummary = document.getElementById('tomorrowSummary');
    const outlookEmoji = document.getElementById('outlookEmoji');
    const outlookText = document.getElementById('outlookText');
    const recoveryReasoning = document.getElementById('recoveryReasoning');
    
    const outlook = getTomorrowOutlook(predictedScore);
    if (tomorrowSummary) tomorrowSummary.textContent = outlook.summary;
    if (outlookEmoji) outlookEmoji.textContent = outlook.emoji;
    if (outlookText) outlookText.textContent = outlook.feeling;
    if (recoveryReasoning) recoveryReasoning.textContent = outlook.reasoning;
    if (elements.recoveryRecommendation) elements.recoveryRecommendation.textContent = recovery.recommendation || outlook.tip;
    
    // Activity Suggestion (Strain) - Friendly Version
    const strain = predictions.strain || {};
    const optimalStrain = strain.optimal_strain || 10;
    
    if (elements.predictedStrain) elements.predictedStrain.textContent = optimalStrain;
    
    const activitySummary = document.getElementById('activitySummary');
    const activity = getActivitySuggestion(optimalStrain, strain.current_recovery || 70);
    if (activitySummary) activitySummary.textContent = activity.summary;
    if (elements.strainRecommendation) elements.strainRecommendation.textContent = strain.activity_recommendation || activity.tip;
    
    // Workout suggestions - More friendly format
    const suggestions = strain.workout_suggestions || [];
    if (elements.workoutSuggestions) {
        elements.workoutSuggestions.innerHTML = suggestions.map(w => `
            <div class="workout-suggestion">
                <span class="workout-type">${getActivityEmoji(w.type)} ${w.type}</span>
                <span class="workout-details">
                    <span>${w.duration}</span>
                    <span>•</span>
                    <span>${w.intensity} intensity</span>
                </span>
            </div>
        `).join('');
    }
    
    // Health Check (Risk) - Friendly Version
    const risk = predictions.risk || {};
    const riskScore = Math.round(risk.risk_score || 0);
    const riskLevel = risk.risk_level || 'low';
    
    if (elements.riskScore) elements.riskScore.textContent = riskScore;
    
    const healthEmoji = document.getElementById('healthEmoji');
    const healthText = document.getElementById('healthText');
    const healthTips = document.getElementById('healthTips');
    
    const healthCheck = getHealthCheckMessage(riskScore, riskLevel, risk.risk_factors || []);
    if (healthEmoji) healthEmoji.textContent = healthCheck.emoji;
    if (healthText) healthText.textContent = healthCheck.message;
    
    if (healthTips) {
        healthTips.innerHTML = healthCheck.tips.map(tip => `
            <div class="health-tip">
                <span class="health-tip-icon">${tip.icon}</span>
                <span>${tip.text}</span>
            </div>
        `).join('');
    }
    
    // Technical factors (hidden by default)
    const factors = risk.risk_factors || [];
    if (elements.riskFactors) {
        elements.riskFactors.innerHTML = factors.map(f => `
            <div class="risk-factor-item ${f.severity}">
                <span>${f.factor}</span>
                <span style="color: var(--text-muted)">${f.value}</span>
            </div>
        `).join('') || '<p style="color: var(--text-muted); font-size: 0.85rem;">Nothing to worry about</p>';
    }
}

// Helper functions for friendly predictions
function getTomorrowOutlook(score) {
    if (score >= 80) {
        return {
            summary: "Tomorrow is looking great! You'll likely wake up feeling refreshed and ready for anything.",
            emoji: "🌟",
            feeling: "Feeling Fantastic",
            reasoning: "Your recent sleep quality and recovery patterns suggest an excellent day ahead.",
            tip: "Great time to tackle challenging goals!"
        };
    } else if (score >= 60) {
        return {
            summary: "Tomorrow should be a solid day. You'll have good energy for most activities.",
            emoji: "😊",
            feeling: "Feeling Good",
            reasoning: "Based on your patterns, you should feel pretty good but might want to pace yourself.",
            tip: "A balanced day with moderate activity is ideal."
        };
    } else if (score >= 40) {
        return {
            summary: "Tomorrow might be a slower day. Consider scheduling lighter activities.",
            emoji: "😐",
            feeling: "Taking It Easy",
            reasoning: "Your body shows signs of needing extra recovery time.",
            tip: "Light activity and extra rest will help."
        };
    } else {
        return {
            summary: "Your body needs rest. Tomorrow would be a great day to recharge.",
            emoji: "🛋️",
            feeling: "Rest Day",
            reasoning: "Your metrics suggest prioritizing recovery over exertion.",
            tip: "Focus on sleep, nutrition, and relaxation."
        };
    }
}

function getActivitySuggestion(strain, recovery) {
    if (strain >= 15) {
        return {
            summary: "Your body can handle a challenging workout today. Go for it!",
            tip: "High-intensity activities will feel great today."
        };
    } else if (strain >= 10) {
        return {
            summary: "A moderate workout is perfect for today. Mix things up!",
            tip: "Try a mix of cardio and strength training."
        };
    } else if (strain >= 5) {
        return {
            summary: "Stick to gentle movement today. Your body will thank you.",
            tip: "Walking, yoga, or stretching are great choices."
        };
    } else {
        return {
            summary: "Today's a rest day. Light stretching is all you need.",
            tip: "Skip the gym - recovery is your workout today."
        };
    }
}

function getActivityEmoji(type) {
    const emojis = {
        'HIIT': '🔥', 'CrossFit': '💪', 'Running': '🏃', 'Cycling': '🚴',
        'Swimming': '🏊', 'Yoga': '🧘', 'Pilates': '🤸', 'Walking': '🚶',
        'Stretching': '🙆', 'Meditation': '🧘', 'Strength Training': '🏋️',
        'Light Jog': '🏃'
    };
    return emojis[type] || '🏃';
}

function getHealthCheckMessage(riskScore, level, factors) {
    if (riskScore < 30) {
        return {
            emoji: "✅",
            message: "All good! Your health metrics look great.",
            tips: [
                { icon: "💪", text: "Keep up your current routine" },
                { icon: "🥗", text: "Your nutrition is supporting your health well" }
            ]
        };
    } else if (riskScore < 60) {
        const tips = [];
        factors.forEach(f => {
            if (f.factor.includes('Sleep')) tips.push({ icon: "😴", text: "Try to get to bed a bit earlier tonight" });
            if (f.factor.includes('HRV')) tips.push({ icon: "🧘", text: "Some relaxation or meditation could help" });
        });
        return {
            emoji: "👀",
            message: "A few things to keep an eye on.",
            tips: tips.length ? tips : [{ icon: "💡", text: "Small adjustments can make a big difference" }]
        };
    } else {
        return {
            emoji: "⚠️",
            message: "Your body is asking for extra care today.",
            tips: [
                { icon: "🛋️", text: "Prioritize rest and recovery" },
                { icon: "🥤", text: "Stay hydrated and eat well" },
                { icon: "😴", text: "An early bedtime would really help" }
            ]
        };
    }
}

// Update chart summaries with trend analysis
function updateChartSummaries(trends) {
    // Recovery/Energy trend
    const recoverySummary = document.getElementById('recoverySummary');
    if (recoverySummary && trends.recovery?.length > 1) {
        const trend = calculateTrendDirection(trends.recovery);
        const avg = Math.round(trends.recovery.reduce((a, b) => a + b, 0) / trends.recovery.length);
        recoverySummary.innerHTML = `
            <span class="summary-icon">${trend.emoji}</span>
            <p><strong>Your energy is ${trend.text}</strong> with an average of ${avg}%. ${trend.message}</p>
        `;
        recoverySummary.className = `chart-summary ${trend.class}`;
    }
    
    // Sleep trend
    const sleepSummary = document.getElementById('sleepSummary');
    if (sleepSummary && trends.sleep_hours?.length > 1) {
        const trend = calculateTrendDirection(trends.sleep_hours);
        const avg = (trends.sleep_hours.reduce((a, b) => a + b, 0) / trends.sleep_hours.length).toFixed(1);
        sleepSummary.innerHTML = `
            <span class="summary-icon">${trend.emoji}</span>
            <p><strong>You're averaging ${avg} hours of sleep</strong> - ${trend.sleepMessage}</p>
        `;
        sleepSummary.className = `chart-summary ${trend.class}`;
    }
    
    // HRV/Stress trend
    const hrvSummary = document.getElementById('hrvSummary');
    if (hrvSummary && trends.hrv?.length > 1) {
        const trend = calculateTrendDirection(trends.hrv);
        const stressLevel = trend.direction > 0 ? 'decreasing' : trend.direction < 0 ? 'increasing' : 'stable';
        hrvSummary.innerHTML = `
            <span class="summary-icon">${trend.direction > 0 ? '😌' : trend.direction < 0 ? '😟' : '🙂'}</span>
            <p><strong>Your stress levels are ${stressLevel}</strong> - ${trend.direction >= 0 ? 'great job staying calm!' : 'try some relaxation techniques.'}</p>
        `;
        hrvSummary.className = `chart-summary ${trend.direction >= 0 ? 'improving' : 'declining'}`;
    }
    
    // Calories trend
    const caloriesSummary = document.getElementById('caloriesSummary');
    if (caloriesSummary && trends.calories?.length > 1) {
        const total = Math.round(trends.calories.reduce((a, b) => a + b, 0));
        const avg = Math.round(total / trends.calories.length);
        caloriesSummary.innerHTML = `
            <span class="summary-icon">🔥</span>
            <p><strong>You've burned ${total.toLocaleString()} calories</strong> (about ${avg.toLocaleString()} per day) - keep moving!</p>
        `;
    }
}

function calculateTrendDirection(data) {
    if (data.length < 2) return { direction: 0, text: 'stable', emoji: '📊', class: '', message: '', sleepMessage: '' };
    
    const firstHalf = data.slice(0, Math.floor(data.length / 2));
    const secondHalf = data.slice(Math.floor(data.length / 2));
    
    const firstAvg = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
    const secondAvg = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;
    
    const change = ((secondAvg - firstAvg) / firstAvg) * 100;
    
    if (change > 5) {
        return { 
            direction: 1, 
            text: 'improving', 
            emoji: '📈', 
            class: 'improving',
            message: 'Keep up the good work!',
            sleepMessage: "that's getting better!"
        };
    } else if (change < -5) {
        return { 
            direction: -1, 
            text: 'declining', 
            emoji: '📉', 
            class: 'declining',
            message: 'Let\'s work on getting this back up.',
            sleepMessage: 'try to get more rest.'
        };
    } else {
        return { 
            direction: 0, 
            text: 'staying steady', 
            emoji: '📊', 
            class: '',
            message: 'Consistent is good!',
            sleepMessage: 'staying consistent.'
        };
    }
}

function updateInsights(correlations) {
    const insights = correlations?.insights || [];
    
    if (insights.length === 0) {
        elements.insightsGrid.innerHTML = `
            <div class="insight-card">
                <div class="insight-title">📊 Building Insights...</div>
                <p class="insight-description">Continue using the app to generate personalized insights from your health data patterns.</p>
            </div>
        `;
        return;
    }
    
    elements.insightsGrid.innerHTML = insights.map(insight => `
        <div class="insight-card">
            <div class="insight-title">
                ${getInsightIcon(insight.type)} ${insight.title}
                <span class="insight-strength">${Math.round(insight.strength * 100)}%</span>
            </div>
            <p class="insight-description">${insight.description}</p>
        </div>
    `).join('');
}

function getInsightIcon(type) {
    const icons = {
        'sleep_recovery': '😴', 'strain_recovery': '⚡',
        'hrv_recovery': '💓', 'deep_sleep': '🌙'
    };
    return icons[type] || '📊';
}

// ============ Chart Functions ============
function initializeCharts() {
    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
            legend: {
                position: 'top',
                labels: { usePointStyle: true, padding: 20 }
            },
            tooltip: {
                backgroundColor: 'rgba(15, 22, 19, 0.95)',
                padding: 12,
                cornerRadius: 8,
                titleColor: '#f0fdf4',
                bodyColor: '#a7f3d0'
            }
        },
        scales: {
            x: {
                grid: { display: false },
                ticks: { maxTicksLimit: 8 }
            },
            y: {
                grid: { color: 'rgba(74, 222, 128, 0.1)' },
                ticks: { padding: 10 }
            }
        }
    };

    // Recovery & Strain Chart
    const recoveryCtx = document.getElementById('recoveryChart')?.getContext('2d');
    if (recoveryCtx) {
        charts.recovery = new Chart(recoveryCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Recovery %',
                        data: [],
                        borderColor: '#4ade80',
                        backgroundColor: 'rgba(74, 222, 128, 0.1)',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 3
                    },
                    {
                        label: 'Strain',
                        data: [],
                        borderColor: '#fbbf24',
                        backgroundColor: 'transparent',
                        tension: 0.4,
                        pointRadius: 3,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                ...chartOptions,
                scales: {
                    ...chartOptions.scales,
                    y: { ...chartOptions.scales.y, min: 0, max: 100, title: { display: true, text: 'Recovery %' } },
                    y1: { position: 'right', min: 0, max: 21, grid: { display: false }, title: { display: true, text: 'Strain' } }
                }
            }
        });
    }

    // Sleep Chart
    const sleepCtx = document.getElementById('sleepChart')?.getContext('2d');
    if (sleepCtx) {
        charts.sleep = new Chart(sleepCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Deep Sleep',
                        data: [],
                        backgroundColor: '#8b5cf6',
                        borderRadius: 4
                    },
                    {
                        label: 'REM Sleep',
                        data: [],
                        backgroundColor: '#22d3ee',
                        borderRadius: 4
                    },
                    {
                        label: 'Light Sleep',
                        data: [],
                        backgroundColor: 'rgba(167, 243, 208, 0.3)',
                        borderRadius: 4
                    }
                ]
            },
            options: {
                ...chartOptions,
                scales: {
                    ...chartOptions.scales,
                    x: { ...chartOptions.scales.x, stacked: true },
                    y: { ...chartOptions.scales.y, stacked: true, title: { display: true, text: 'Hours' } }
                }
            }
        });
    }

    // HRV Chart
    const hrvCtx = document.getElementById('hrvChart')?.getContext('2d');
    if (hrvCtx) {
        charts.hrv = new Chart(hrvCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'HRV (ms)',
                        data: [],
                        borderColor: '#4ade80',
                        backgroundColor: 'rgba(74, 222, 128, 0.1)',
                        fill: true,
                        tension: 0.4
                    },
                    {
                        label: 'Resting HR',
                        data: [],
                        borderColor: '#f87171',
                        backgroundColor: 'transparent',
                        tension: 0.4,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                ...chartOptions,
                scales: {
                    ...chartOptions.scales,
                    y: { ...chartOptions.scales.y, title: { display: true, text: 'HRV (ms)' } },
                    y1: { position: 'right', grid: { display: false }, title: { display: true, text: 'Heart Rate (bpm)' } }
                }
            }
        });
    }

    // Calories Chart
    const caloriesCtx = document.getElementById('caloriesChart')?.getContext('2d');
    if (caloriesCtx) {
        charts.calories = new Chart(caloriesCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Calories Burned',
                    data: [],
                    backgroundColor: (context) => {
                        const chart = context.chart;
                        const { ctx, chartArea } = chart;
                        if (!chartArea) return '#4ade80';
                        const gradient = ctx.createLinearGradient(0, chartArea.bottom, 0, chartArea.top);
                        gradient.addColorStop(0, 'rgba(74, 222, 128, 0.5)');
                        gradient.addColorStop(1, 'rgba(34, 211, 238, 0.8)');
                        return gradient;
                    },
                    borderRadius: 6
                }]
            },
            options: {
                ...chartOptions,
                scales: {
                    ...chartOptions.scales,
                    y: { ...chartOptions.scales.y, title: { display: true, text: 'Calories' } }
                }
            }
        });
    }
}

function updateCharts(trends) {
    if (!trends || !trends.dates || trends.dates.length === 0) return;
    
    // Update chart summaries with trend analysis
    updateChartSummaries(trends);
    
    const labels = trends.dates.map(d => {
        const date = new Date(d);
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    });
    
    // Update Recovery Chart
    if (charts.recovery) {
        charts.recovery.data.labels = labels;
        charts.recovery.data.datasets[0].data = trends.recovery;
        charts.recovery.data.datasets[1].data = trends.strain;
        charts.recovery.update('none');
    }
    
    // Update Sleep Chart
    if (charts.sleep) {
        // Calculate light sleep
        const lightSleep = trends.sleep_hours.map((total, i) => {
            const deep = trends.deep_sleep[i] || 0;
            const rem = trends.rem_sleep[i] || 0;
            return Math.max(0, total - deep - rem);
        });
        
        charts.sleep.data.labels = labels;
        charts.sleep.data.datasets[0].data = trends.deep_sleep;
        charts.sleep.data.datasets[1].data = trends.rem_sleep;
        charts.sleep.data.datasets[2].data = lightSleep;
        charts.sleep.update('none');
    }
    
    // Update HRV Chart
    if (charts.hrv) {
        charts.hrv.data.labels = labels;
        charts.hrv.data.datasets[0].data = trends.hrv;
        charts.hrv.data.datasets[1].data = trends.resting_hr;
        charts.hrv.update('none');
    }
    
    // Update Calories Chart
    if (charts.calories) {
        charts.calories.data.labels = labels;
        charts.calories.data.datasets[0].data = trends.calories;
        charts.calories.update('none');
    }
}

// ============ UI Component State (Learning System) ============
const componentStates = new Map();
const ComponentState = { PENDING: 'pending', SUBMITTING: 'submitting', SUBMITTED: 'submitted', ERROR: 'error' };

function escapeHtml(str) {
    if (str == null) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function renderUIComponent(component) {
    const componentId = `${component.question_id}_${Date.now()}`;
    componentStates.set(componentId, { state: ComponentState.PENDING, selectedOptions: [] });

    const container = document.createElement('div');
    container.className = `ui-component ${component.type}`;
    container.dataset.componentId = componentId;
    container.dataset.questionId = component.question_id;
    container.dataset.category = component.category;

    container.innerHTML = `<p class="ui-prompt">${escapeHtml(component.prompt)}</p>`;
    const optionsDiv = document.createElement('div');
    optionsDiv.className = 'ui-options';

    const isMulti = component.type === 'multi_select';
    const inputType = isMulti ? 'checkbox' : 'radio';
    (component.options || []).forEach(opt => {
        const label = document.createElement('label');
        label.className = 'option-label';
        label.dataset.optionId = escapeHtml(opt.id);
        label.innerHTML = `
            <input type="${inputType}" name="${componentId}" value="${escapeHtml(opt.id)}">
            <span class="option-content">
                ${opt.emoji ? `<span class="option-emoji">${escapeHtml(opt.emoji)}</span>` : ''}
                <span class="option-text">${escapeHtml(opt.label)}</span>
            </span>
            <span class="option-check" aria-hidden="true">✓</span>
        `;
        optionsDiv.appendChild(label);
    });
    container.appendChild(optionsDiv);

    const submitBtn = document.createElement('button');
    submitBtn.type = 'button';
    submitBtn.className = 'ui-submit-btn';
    submitBtn.textContent = 'Continue';
    submitBtn.onclick = () => handleComponentSubmit(componentId, container);
    container.appendChild(submitBtn);

    if (component.allow_skip !== false) {
        const skipBtn = document.createElement('button');
        skipBtn.type = 'button';
        skipBtn.className = 'ui-skip-btn';
        skipBtn.textContent = 'Skip';
        skipBtn.onclick = () => handleComponentSkip(componentId, container);
        container.appendChild(skipBtn);
    }

    return container;
}

async function fetchWithRetry(url, options, maxRetries) {
    let lastErr;
    for (let i = 0; i < maxRetries; i++) {
        try {
            const res = await fetch(url, options);
            return await res.json();
        } catch (e) {
            lastErr = e;
            if (i < maxRetries - 1) await new Promise(r => setTimeout(r, 1000 * (i + 1)));
        }
    }
    throw lastErr;
}

async function handleComponentSubmit(componentId, container) {
    const state = componentStates.get(componentId);
    if (!state || state.state !== ComponentState.PENDING) return;

    const inputs = container.querySelectorAll('input:checked');
    const selectedOptions = Array.from(inputs).map(i => i.value);
    if (selectedOptions.length === 0) {
        showNotification('Please select at least one option', 'info');
        return;
    }

    state.state = ComponentState.SUBMITTING;
    state.selectedOptions = selectedOptions;
    container.classList.add('submitting');

    const questionId = container.dataset.questionId;
    const category = container.dataset.category;

    try {
        const data = await fetchWithRetry(
            `${API_BASE}/users/${currentUserId}/preferences`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question_id: questionId,
                    category: category,
                    selected_options: selectedOptions,
                    skipped: false
                })
            },
            3
        );

        if (data.status === 'ok') {
            state.state = ComponentState.SUBMITTED;
            container.classList.remove('submitting');
            container.classList.add('submitted');
            showNotification('Preference saved!', 'success');
            addSystemMessage(`Got it! I'll remember your ${category} preferences.`);
        } else {
            throw new Error(data.message || 'Failed to save');
        }
    } catch (error) {
        console.error('Preference save error:', error);
        state.state = ComponentState.ERROR;
        container.classList.remove('submitting');
        container.classList.add('error');
        showNotification('Failed to save preference. Please try again.', 'error');
        const retryBtn = document.createElement('button');
        retryBtn.type = 'button';
        retryBtn.className = 'ui-retry-btn';
        retryBtn.textContent = 'Retry';
        retryBtn.onclick = () => {
            container.classList.remove('error');
            retryBtn.remove();
            state.state = ComponentState.PENDING;
            handleComponentSubmit(componentId, container);
        };
        container.appendChild(retryBtn);
    }
}

async function handleComponentSkip(componentId, container) {
    const state = componentStates.get(componentId);
    if (!state || state.state !== ComponentState.PENDING) return;

    state.state = ComponentState.SUBMITTING;
    container.classList.add('submitting');

    const questionId = container.dataset.questionId;
    const category = container.dataset.category;

    try {
        const data = await fetchWithRetry(
            `${API_BASE}/users/${currentUserId}/preferences`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question_id: questionId,
                    category: category,
                    selected_options: [],
                    skipped: true
                })
            },
            3
        );

        if (data.status === 'ok') {
            state.state = ComponentState.SUBMITTED;
            container.classList.remove('submitting');
            container.classList.add('submitted');
            addSystemMessage(`Skipped. You can share ${category} preferences anytime.`);
        } else {
            throw new Error(data.message || 'Failed to save');
        }
    } catch (error) {
        console.error('Preference skip error:', error);
        state.state = ComponentState.ERROR;
        container.classList.remove('submitting');
        showNotification('Something went wrong. Please try again.', 'error');
    }
}

// ============ Message Functions ============
function addMessage(content, type) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const name = type === 'user' ? 'You' : 'WellnessAI';
    
    const avatarContent = type === 'assistant' ? `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
        </svg>
    ` : '👤';
    
    messageDiv.innerHTML = `
        <div class="message-avatar" aria-hidden="true">${avatarContent}</div>
        <div class="message-content">
            <div class="message-header">
                <span class="message-name">${name}</span>
                <span class="message-time">${time}</span>
            </div>
            <div class="message-text">${formatMessage(content)}</div>
        </div>
    `;
    
    elements.chatMessages.appendChild(messageDiv);
    scrollToBottom();
    return messageDiv;
}

function addSystemMessage(content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message system';
    messageDiv.style.cssText = 'justify-content: center; padding: 8px 0;';
    messageDiv.innerHTML = `
        <div style="background: var(--bg-tertiary); border: 1px solid var(--border-color); border-radius: var(--radius-full); padding: 8px 20px; font-size: 0.8rem; color: var(--text-muted);">
            ${content}
        </div>
    `;
    elements.chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

function formatMessage(content) {
    let formatted = content
        .replace(/^### (.*?)$/gm, '<h4>$1</h4>')
        .replace(/^## (.*?)$/gm, '<h3>$1</h3>')
        .replace(/^# (.*?)$/gm, '<h2>$1</h2>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code>$1</code>')
        .replace(/^---$/gm, '<hr>')
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>');
    
    formatted = formatted.replace(/(<br>)?- (.*?)(<br>|$)/g, '<li>$2</li>');
    formatted = formatted.replace(/(<li>.*?<\/li>)+/g, '<ul>$&</ul>');
    formatted = formatted.replace(/<\/ul><br><ul>/g, '');
    formatted = formatted.replace(/<br><ul>/g, '<ul>');
    formatted = formatted.replace(/<\/ul><br>/g, '</ul>');
    
    if (!formatted.startsWith('<') || formatted.startsWith('<strong>') || formatted.startsWith('<em>')) {
        formatted = `<p>${formatted}</p>`;
    }
    
    formatted = formatted.replace(/<p><\/p>/g, '');
    formatted = formatted.replace(/<p><br><\/p>/g, '');
    
    return formatted;
}

function addTypingIndicator() {
    const id = 'typing-' + Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.id = id;
    
    messageDiv.innerHTML = `
        <div class="message-avatar" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
            </svg>
        </div>
        <div class="message-content">
            <div class="message-header">
                <span class="message-name">WellnessAI</span>
                <span class="message-time">thinking...</span>
            </div>
            <div class="typing-indicator" role="status" aria-label="WellnessAI is typing">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;
    
    elements.chatMessages.appendChild(messageDiv);
    scrollToBottom();
    return id;
}

function removeTypingIndicator(id) {
    const element = document.getElementById(id);
    if (element) {
        element.style.animation = 'messageOut 0.2s ease forwards';
        setTimeout(() => element.remove(), 200);
    }
}

function scrollToBottom() {
    elements.chatContainer.scrollTo({ top: elements.chatContainer.scrollHeight, behavior: 'smooth' });
}

// Loading & Notifications
function showLoadingState(show) {
    if (show) {
        elements.loadingOverlay.classList.add('visible');
    } else {
        elements.loadingOverlay.classList.remove('visible');
    }
}

function showNotification(message, type = 'info') {
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed; bottom: 100px; left: 50%; transform: translateX(-50%) translateY(20px);
        background: ${type === 'error' ? 'var(--accent-danger)' : type === 'success' ? 'var(--accent-success)' : 'var(--bg-elevated)'};
        color: ${type === 'info' ? 'var(--text-primary)' : 'var(--bg-primary)'};
        padding: 12px 24px; border-radius: var(--radius-full); font-size: 0.9rem; font-weight: 500;
        z-index: 1001; opacity: 0; transition: all 0.3s ease; box-shadow: var(--shadow-lg);
    `;
    toast.textContent = message;
    document.body.appendChild(toast);
    
    requestAnimationFrame(() => {
        toast.style.opacity = '1';
        toast.style.transform = 'translateX(-50%) translateY(0)';
    });
    
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(-50%) translateY(20px)';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// CSS for message out animation
const style = document.createElement('style');
style.textContent = `@keyframes messageOut { to { opacity: 0; transform: translateY(-10px); } }`;
document.head.appendChild(style);

// Export for debugging
window.WellnessAI = { loadUsers, loadUserData, loadWeather, loadDashboardData, currentUserId: () => currentUserId, healthRisks: () => healthRisks, charts };
