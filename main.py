import os
from nicegui import ui

# 1. The Home Route (The Spoke's Landing Page)
@ui.page('/')
def home():
    ui.label('Mahini & Maher Live Models').classes('text-2xl font-bold mb-4')
    ui.link('Black-Litterman (BL)', '/bl').classes('block mb-2 text-blue-500')
    ui.link('Hidden Markov Model (HMM)', '/hmm').classes('block mb-2 text-blue-500')
    ui.link('Heston Stochastic Volatility (HSV)', '/hsv').classes('block mb-2 text-blue-500')

# 2. The Black-Litterman Route
@ui.page('/bl')
def bl_page():
    ui.label('Black-Litterman Model Application').classes('text-xl font-bold mb-4')
    ui.label('Configure your market views and asset inputs below:')
    
    def run_bl():
        ui.notify('Executing Black-Litterman optimization calculation...')

    ui.button('Run BL Optimization', on_click=run_bl).classes('bg-blue-500 text-white mt-4')
    ui.link('← Back to Home', '/').classes('block mt-6 text-gray-500')

# 3. The Hidden Markov Model Route
@ui.page('/hmm')
def hmm_page():
    ui.label('Hidden Markov Model Application').classes('text-xl font-bold mb-4')
    ui.label('Initialize historical return data for regime classification:')
    
    def run_hmm():
        ui.notify('Fitting Hidden Markov Model states...')

    ui.button('Run HMM Regime Detection', on_click=run_hmm).classes('bg-green-500 text-white mt-4')
    ui.link('← Back to Home', '/').classes('block mt-6 text-gray-500')

# 4. The Heston Stochastic Volatility Route
@ui.page('/hsv')
def hsv_page():
    ui.label('Heston Stochastic Volatility Application').classes('text-xl font-bold mb-4')
    ui.label('Calibrate volatility parameters and pricing surface:')
    
    def run_hsv():
        ui.notify('Executing Heston Stochastic Volatility calibration...')

    ui.button('Run HSV Calibration', on_click=run_hsv).classes('bg-orange-500 text-white mt-4')
    ui.link('← Back to Home', '/').classes('block mt-6 text-gray-500')

# 5. Dynamic Server Configuration for Render
port = int(os.environ.get('PORT', 10000))
ui.run(host='0.0.0.0', port=port, reload=False)
