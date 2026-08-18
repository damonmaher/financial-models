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
    ui.label('Black-Litterman Model Application').classes('text-xl')
    ui.label('Live Python logic goes here.')

# 3. The Hidden Markov Model Route
@ui.page('/hmm')
def hmm_page():
    ui.label('Hidden Markov Model Application').classes('text-xl')
    ui.label('Live Python logic goes here.')

# 4. The Heston Stochastic Volatility Route
@ui.page('/hsv')
def hsv_page():
    ui.label('Heston Stochastic Volatility Application').classes('text-xl')
    ui.label('Live Python logic goes here.')

# 5. Dynamic Server Configuration for Render
port = int(os.environ.get('PORT', 10000))
ui.run(host='0.0.0.0', port=port, reload=False)
