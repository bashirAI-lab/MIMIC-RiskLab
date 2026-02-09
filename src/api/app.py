from flask import Flask, request, jsonify, render_template_string
import joblib
import os
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load Model and Scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# src/api -> src/models (up one level to src, then models)
models_dir = os.path.join(os.path.dirname(BASE_DIR), 'models')
MODEL_PATH = os.path.join(models_dir, 'model.joblib')
SCALER_PATH = os.path.join(models_dir, 'scaler.pkl')

print(f"Loading model from {MODEL_PATH}...")
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Scikit-Learn Model loaded successfully.")
    else:
        print(f"CRITICAL ERROR: Model file not found at {MODEL_PATH}")
        model = None
        scaler = None
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load model/scaler. {e}")
    model = None
    scaler = None

# Feature columns verification (must match training)
FEATURE_COLS = ['age', 'gender', 'lab_count', 'abnormal_count', 
                'type_ELECTIVE', 'type_EMERGENCY', 'type_URGENT']

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MIMIC-III Clinical Diagnostic System</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <!-- Lucide Icons -->
    <script src="https://unpkg.com/lucide@latest"></script>
    <style>
        .loader {
            border-top-color: #0ea5e9;
            -webkit-animation: spinner 1.5s linear infinite;
            animation: spinner 1.5s linear infinite;
        }

        @keyframes spinner {
            0% {
                transform: rotate(0deg);
            }

            100% {
                transform: rotate(360deg);
            }
        }
    </style>
</head>

<body class="bg-slate-50 text-slate-800 font-sans antialiased h-screen flex overflow-hidden">

    <!-- Sidebar -->
    <aside class="w-64 bg-slate-900 text-white flex flex-col shadow-xl z-10">
        <div class="h-16 flex items-center px-6 border-b border-slate-700 bg-slate-950">
            <i data-lucide="activity" class="w-8 h-8 text-medical-500 mr-3"></i>
            <span class="text-xl font-bold tracking-tight">MIMIC-III Setup</span>
        </div>

        <nav class="flex-1 py-6 px-3 space-y-1">
            <a href="#"
                class="flex items-center px-3 py-3 rounded-lg bg-medical-600 text-white shadow-lg transition-transform transform scale-105">
                <i data-lucide="user-plus" class="w-5 h-5 mr-3"></i>
                <span class="font-medium">Patient Diagnosis</span>
            </a>
            <a href="#" id="btn-analytics"
                class="flex items-center px-3 py-3 rounded-lg text-slate-400 hover:bg-slate-800 hover:text-white transition-colors">
                <i data-lucide="bar-chart-2" class="w-5 h-5 mr-3"></i>
                <span class="font-medium">Analytics Dashboard</span>
            </a>
            <a href="#" id="btn-logs"
                class="flex items-center px-3 py-3 rounded-lg text-slate-400 hover:bg-slate-800 hover:text-white transition-colors">
                <i data-lucide="history" class="w-5 h-5 mr-3"></i>
                <span class="font-medium">Historical Logs</span>
            </a>
            <a href="#" id="btn-settings"
                class="flex items-center px-3 py-3 rounded-lg text-slate-400 hover:bg-slate-800 hover:text-white transition-colors">
                <i data-lucide="settings" class="w-5 h-5 mr-3"></i>
                <span class="font-medium">Settings</span>
            </a>
        </nav>

        <div class="p-4 border-t border-slate-700">
            <div class="flex items-center">
                <div class="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center text-xs font-bold">DR
                </div>
                <div class="ml-3">
                    <p class="text-sm font-medium">Dr. Admin</p>
                    <p class="text-xs text-slate-500">Cardiology Dept.</p>
                </div>
            </div>
        </div>
    </aside>

    <!-- Main Content -->
    <main class="flex-1 flex flex-col relative overflow-y-auto">
        <!-- Header -->
        <header class="bg-white h-16 border-b border-slate-200 flex items-center justify-between px-8 shadow-sm">
            <h1 class="text-2xl font-bold text-slate-800">New Patient Assessment</h1>
            <div class="flex items-center space-x-4">
                <span class="text-sm text-slate-500">System Status: <span
                        class="text-green-500 font-semibold">Online</span></span>
                <button class="p-2 rounded-full hover:bg-slate-100 text-slate-600">
                    <i data-lucide="bell" class="w-5 h-5"></i>
                </button>
            </div>
        </header>

        <div class="flex-1 p-8 grid grid-cols-1 lg:grid-cols-2 gap-8">

            <!-- Input Form -->
            <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-6 h-fit">
                <div class="flex items-center justify-between mb-6">
                    <h2 class="text-lg font-semibold text-slate-700 flex items-center">
                        <i data-lucide="clipboard-list" class="w-5 h-5 mr-2 text-medical-600"></i>
                        Clinical Inputs
                    </h2>
                    <span class="text-xs font-medium text-slate-400 bg-slate-100 px-2 py-1 rounded">MIMIC-III
                        Data</span>
                </div>

                <form id="diagnosisForm" class="space-y-5">
                    <div class="grid grid-cols-2 gap-5">
                        <div class="space-y-2">
                            <label class="block text-sm font-medium text-slate-700">Age</label>
                            <input type="number" id="age" value="65"
                                class="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-medical-500 focus:border-medical-500 outline-none transition"
                                placeholder="Years">
                        </div>
                        <div class="space-y-2">
                            <label class="block text-sm font-medium text-slate-700">Gender</label>
                            <select id="gender"
                                class="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-medical-500 focus:border-medical-500 outline-none transition bg-white">
                                <option value="M">Male</option>
                                <option value="F">Female</option>
                            </select>
                        </div>
                    </div>

                    <div class="space-y-2">
                        <label class="block text-sm font-medium text-slate-700">Admission Type</label>
                        <select id="admission_type"
                            class="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-medical-500 focus:border-medical-500 outline-none transition bg-white">
                            <option value="EMERGENCY">Emergency</option>
                            <option value="URGENT">Urgent</option>
                            <option value="ELECTIVE">Elective</option>
                        </select>
                    </div>

                    <div class="grid grid-cols-2 gap-5">
                        <div class="space-y-2">
                            <label class="block text-sm font-medium text-slate-700">Total Lab Events</label>
                            <input type="number" id="lab_count" value="40"
                                class="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-medical-500 focus:border-medical-500 outline-none transition"
                                placeholder="Count">
                            <p class="text-xs text-slate-500">Normal Range: 20 - 60</p>
                        </div>
                        <div class="space-y-2">
                            <label class="block text-sm font-medium text-slate-700">Abnormal Labs</label>
                            <input type="number" id="abnormal_count" value="5"
                                class="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-red-500 outline-none transition"
                                placeholder="Count">
                            <p class="text-xs text-slate-500">Critical items flagged</p>
                        </div>
                    </div>

                    <div class="pt-6">
                        <button type="submit"
                            class="w-full bg-medical-600 hover:bg-medical-700 text-white font-bold py-3 rounded-lg shadow-lg flex items-center justify-center transition-all bg-gradient-to-r from-cyan-600 to-blue-600">
                            <i data-lucide="activity" class="w-5 h-5 mr-2"></i>
                            Run Diagnostic Model
                        </button>
                    </div>
                </form>
            </div>

            <!-- Results Panel -->
            <div class="space-y-8">
                <!-- Result Card -->
                <div
                    class="bg-white rounded-xl shadow-sm border border-slate-200 p-6 relative overflow-hidden min-h-[300px] flex flex-col justify-center items-center">

                    <!-- Loading Overlay -->
                    <div id="loader"
                        class="absolute inset-0 bg-white/90 backdrop-blur-sm z-10 flex flex-col items-center justify-center hidden">
                        <div
                            class="loader ease-linear rounded-full border-4 border-t-4 border-slate-200 h-12 w-12 mb-4">
                        </div>
                        <p class="text-slate-600 font-medium animate-pulse">Analyzing Patient Vitals...</p>
                    </div>

                    <!-- Chart Container -->
                    <div class="w-full max-w-xs relative mb-4">
                        <canvas id="gaugeChart"></canvas>
                        <div class="absolute inset-0 flex items-center justify-center flex-col top-10">
                            <span id="scoreValues" class="text-4xl font-extrabold text-slate-800">--%</span>
                            <span class="text-xs text-slate-500 uppercase tracking-widest mt-1">Mortality Risk</span>
                        </div>
                    </div>

                    <div id="status-badge"
                        class="px-4 py-2 rounded-full bg-slate-100 text-slate-500 font-bold text-sm tracking-wide mb-4">
                        AWAITING INPUT
                    </div>

                </div>

                <!-- AI Explainer / Clinical Justification -->
                <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
                    <h3 class="text-md font-bold text-slate-800 mb-3 flex items-center">
                        <i data-lucide="brain-circuit" class="w-5 h-5 mr-2 text-purple-600"></i>
                        Clinical Justification
                    </h3>
                    <div id="justification" class="text-slate-600 text-sm leading-relaxed">
                        <p>Waiting for diagnostic inputs. The AI model evaluates <strong>Age</strong>, <strong>Admission
                                Type</strong>, and <strong>Lab Abnormalities</strong> to estimate mortality risk during
                            hospital stay.</p>
                    </div>
                </div>
            </div>

        </div>
    </main>

    <script>
        // Global variables
        let gaugeChart;

        // Tailwind Configuration
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        medical: {
                            50: '#f0f9ff',
                            100: '#e0f2fe',
                            500: '#0ea5e9',
                            600: '#0284c7',
                            800: '#075985',
                            900: '#0c4a6e',
                        }
                    }
                }
            }
        };

        window.onload = function () {
            // Initialize Icons
            lucide.createIcons();

            // Chart Configuration
            const ctx = document.getElementById('gaugeChart').getContext('2d');
            gaugeChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['Risk', 'Safe'],
                    datasets: [{
                        data: [0, 100],
                        backgroundColor: ['#e2e8f0', '#e2e8f0'],
                        borderWidth: 0,
                        circumference: 180,
                        rotation: 270,
                        cutout: '85%'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false }, tooltip: { enabled: false } },
                    animation: { animateRotate: true, animateScale: true }
                }
            });

            // Sidebar Button Handlers
            const restrictedMsg = "Access Restricted: This feature is currently under development for the MIMIC-III Clinical Suite.";
            const settingsMsg = "System Configuration Locked: Administrative privileges required to modify risk thresholds.";

            ['btn-analytics', 'btn-logs'].forEach(id => {
                const btn = document.getElementById(id);
                if (btn) {
                    btn.addEventListener('click', (e) => {
                        e.preventDefault();
                        alert(restrictedMsg);
                    });
                }
            });

            const settingsBtn = document.getElementById('btn-settings');
            if (settingsBtn) {
                settingsBtn.addEventListener('click', (e) => {
                    e.preventDefault();
                    alert(settingsMsg);
                });
            }

            // Form Submission
            document.getElementById('diagnosisForm').addEventListener('submit', async (e) => {
                e.preventDefault();

                // Show Loader
                const loader = document.getElementById('loader');
                loader.classList.remove('hidden');

                // Gather Data
                const formData = {
                    age: parseFloat(document.getElementById('age').value),
                    gender: document.getElementById('gender').value,
                    admission_type: document.getElementById('admission_type').value,
                    lab_count: parseInt(document.getElementById('lab_count').value),
                    abnormal_count: parseInt(document.getElementById('abnormal_count').value)
                };

                try {
                    // Simulate network delay for UX (optional, but requested for "Processing" feel)
                    await new Promise(r => setTimeout(r, 800));

                    const response = await fetch('/diagnose', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(formData)
                    });

                    if (!response.ok) throw new Error('Network response was not ok');

                    const result = await response.json();
                    updateDashboard(result);

                } catch (error) {
                    console.error('Error:', error);
                    alert('Diagnostic Service Failed: ' + error.message);
                } finally {
                    loader.classList.add('hidden');
                }
            });
        };

        function updateDashboard(data) {
            const risk = data.mortality_risk; // 0 to 1
            const percentage = (risk * 100).toFixed(1);

            // Update Text
            document.getElementById('scoreValues').innerText = `${percentage}%`;

            // Update Badge and Gauge Color based on three-tier system
            const badge = document.getElementById('status-badge');
            let color = '#22c55e'; // Default Green (LOW)
            
            if (percentage > 55) {
                // CRITICAL RISK
                badge.className = "px-6 py-2 rounded-full bg-red-100 text-red-700 font-bold text-sm tracking-wide mb-4 border border-red-200 shadow-sm";
                badge.innerHTML = `<i data-lucide="alert-triangle" class="inline w-4 h-4 mr-2"></i> CRITICAL RISK`;
                color = '#ef4444'; 
            } else if (percentage >= 35) {
                // MODERATE RISK
                badge.className = "px-6 py-2 rounded-full bg-amber-100 text-amber-700 font-bold text-sm tracking-wide mb-4 border border-amber-200 shadow-sm";
                badge.innerHTML = `<i data-lucide="alert-circle" class="inline w-4 h-4 mr-2"></i> MODERATE RISK`;
                color = '#f59e0b';
            } else {
                // LOW RISK
                badge.className = "px-6 py-2 rounded-full bg-green-100 text-green-700 font-bold text-sm tracking-wide mb-4 border border-green-200 shadow-sm";
                badge.innerHTML = `<i data-lucide="check-circle" class="inline w-4 h-4 mr-2"></i> STABLE`;
                color = '#22c55e';
            }
            
            lucide.createIcons(); // Refresh icons in badge

            gaugeChart.data.datasets[0].data = [percentage, 100 - percentage];
            gaugeChart.data.datasets[0].backgroundColor = [color, '#e2e8f0'];
            gaugeChart.update();

            // Update Justification
            const justificationPanel = document.getElementById('justification');
            const type = document.getElementById('admission_type').value;
            const abnormals = document.getElementById('abnormal_count').value;

            const accuracyFooter = `
                <div class="mt-4 pt-3 border-t border-slate-100 flex justify-between items-center text-[10px] text-slate-400">
                    <span>MIMIC-III Engine v1.0.2</span>
                    <span class="font-medium">Model Accuracy: 80.7%</span>
                </div>`;

            if (percentage > 55) {
                let labWarning = abnormals > 10 ?
                    `<li class="text-red-700"><strong>High number of abnormal lab results detected, immediate intervention required.</strong></li>` :
                    `<li><strong>Abnormal Labs (${abnormals}):</strong> High deviation suggests systemic instability.</li>`;

                justificationPanel.innerHTML = `
                    <p class="font-medium text-red-800 mb-2">Primary Drivers for High Risk:</p>
                    <ul class="list-disc pl-5 space-y-1">
                        ${labWarning}
                        <li><strong>Admission (${type}):</strong> Contributes to acute acuity score.</li>
                    </ul>
                    <div class="mt-3 p-3 bg-red-50 text-red-800 rounded text-xs border border-red-100">
                        <strong>Action:</strong> Escalation to intensive care unit recommended.
                    </div>
                    ${accuracyFooter}
                `;
            } else if (percentage >= 35) {
                justificationPanel.innerHTML = `
                    <p class="font-medium text-amber-800 mb-2">Moderate Risk Indicators:</p>
                    <p class="mb-3 text-slate-700">Patient shows elevated clinical markers; increased monitoring recommended.</p>
                    <ul class="list-disc pl-5 space-y-1">
                        <li><strong>Lab Deviation:</strong> Abnormal results (${abnormals}) are above baseline but not in critical range.</li>
                        <li><strong>Observation:</strong> Regular neurological and cardiovascular checks advised.</li>
                    </ul>
                    <div class="mt-3 p-3 bg-amber-50 text-amber-800 rounded text-xs border border-amber-100">
                        <strong>Action:</strong> Increase frequency of vitals monitoring.
                    </div>
                    ${accuracyFooter}
                `;
            } else {
                justificationPanel.innerHTML = `
                   <p class="font-medium text-green-800 mb-2">Indicators for Stability:</p>
                    <ul class="list-disc pl-5 space-y-1">
                        <li><strong>Lab Profile:</strong> Abnormal count (${abnormals}) is within acceptable variance.</li>
                        <li><strong>Demographics:</strong> Age-adjusted risk is low.</li>
                    </ul>
                    <div class="mt-3 p-3 bg-green-50 text-green-800 rounded text-xs border border-green-100">
                        <strong>Action:</strong> Continue standard monitoring protocols.
                    </div>
                    ${accuracyFooter}
                `;
            }
        }
    </script>
</body>

</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(INDEX_HTML)

@app.route('/diagnose', methods=['POST'])
def diagnose():
    try:
        data = request.json
        print(f"Received data: {data}")
        
        if model is None or scaler is None:
            return jsonify({'status': 'error', 'message': 'Model or Scaler not loaded on server.'}), 500
        
        # Preprocessing
        # 1. Gender mapping
        gender_val = 0 if data.get('gender') == 'M' else 1
        
        # 2. Admission Type One-Hot
        adm_type = data.get('admission_type', 'EMERGENCY')
        type_elective = 1 if adm_type == 'ELECTIVE' else 0
        type_emergency = 1 if adm_type == 'EMERGENCY' else 0
        type_urgent = 1 if adm_type == 'URGENT' else 0
        
        # 3. Create Feature Vector
        features = [
            data.get('age', 60),
            gender_val,
            data.get('lab_count', 0),
            data.get('abnormal_count', 0),
            type_elective,
            type_emergency,
            type_urgent
        ]
        
        # 4. Scale
        features_array = np.array([features])
        features_scaled = scaler.transform(features_array)
        
        # 5. Predict Probabilities using Scikit-Learn
        prediction_probs = model.predict_proba(features_scaled)
        # Class 1 is mortality risk
        mortality_risk = float(prediction_probs[0][1])
        
        return jsonify({
            'status': 'success',
            'mortality_risk': mortality_risk,
            'risk_level': 'High' if mortality_risk > 0.55 else ('Moderate' if mortality_risk >= 0.35 else 'Low')
        })
        
    except Exception as e:
        print(f"Error in diagnose: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
        
    except Exception as e:
        print(f"Error in diagnose: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
