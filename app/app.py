from calendar import month_abbr, month_name
import base64
from datetime import time
import json
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from dashboard_data import filter_data, summarize, totals
from airport_labels import airport_label, load_airport_names

ROOT = Path(__file__).resolve().parents[1]
HERO_PATH = ROOT / 'assets/aviation-hero.jpg'
DATA_PATH = ROOT / 'data/analysis/flight_analysis_2024.parquet'
METADATA_PATH = ROOT / 'data/analysis/flight_analysis_2024.metadata.json'
WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
NAVY = '#14253D'
BURGUNDY = '#922F48'
GOLD = '#C5A47E'

st.set_page_config(page_title='Aviation Signature | Flight Intelligence', page_icon='✈️', layout='wide')
st.markdown('''<style>
.stApp {background: #F5F0E9; color: #514F55; font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;}
/* Reserve space for Streamlit’s fixed 60px toolbar instead of covering it. */
.block-container {max-width:1250px; padding:calc(3.75rem + .5rem) 2rem 2.5rem;}
.stApp .block-container :is(h1,h2,h3,h4) {color:#14253D; letter-spacing:-.035em;}
h2 {font-size:1.65rem !important; font-weight:600 !important;}
h3 {font-size:1.25rem !important; font-weight:600 !important;}
h4 {font-size:1rem !important; font-weight:600 !important;}
p,label,li {font-size:.92rem; line-height:1.65;}
[data-testid="stCaptionContainer"] {color:#615B60;}
[data-testid="stVerticalBlock"] {gap:.85rem;}
[data-testid="stMetric"] {background:white; border:1px solid #E5DDD3;
 border-radius:12px; padding:.85rem 1rem; box-shadow:0 5px 18px #14253D08;}
[data-testid="stMetricLabel"] p {font-size:.77rem; color:#615B60;}
[data-testid="stMetricValue"] {color:#14253D; font-size:clamp(1.2rem,2vw,1.65rem) !important; font-weight:600;}
[data-testid="stForm"] {box-shadow:0 8px 30px #14253D08; background:white; border:1px solid #E5DDD3; border-radius:16px; padding:1.3rem;}
[data-testid="stExpander"] {background:white; border:1px solid #E5DDD3; border-radius:12px;}
[data-testid="stVegaLiteChart"] {background:white; border:1px solid #E5DDD3; border-radius:12px;}
[data-testid="stForm"] [data-testid="stWidgetLabel"] p {color:#14253D; font-weight:550;}
[data-baseweb="input"],[data-baseweb="select"] > div {border-radius:9px; background:#FFFFFF; border-color:#DCD1C5;}
div.stButton > button,div.stFormSubmitButton > button,div.stDownloadButton > button {
 border-radius:9px; min-height:2.6rem; font-size:.88rem;
 transition:background-color .18s ease,border-color .18s ease,box-shadow .18s ease;}
div.stButton > button:hover,div.stDownloadButton > button:hover {
 border-color:#C5A47E; background:#F5F0E9; color:#14253D; box-shadow:0 3px 10px #14253D10;}
div.stButton > button[kind="primary"],div.stFormSubmitButton > button[kind="primary"] {
 background:#922F48; border:1px solid #922F48; color:white;}
div.stButton > button[kind="primary"]:hover,div.stFormSubmitButton > button[kind="primary"]:hover {
 background:#76233A; border-color:#76233A; box-shadow:0 4px 12px #922F4820;}
.st-key-site_navigation {padding:.25rem 0 1rem; border-bottom:1px solid #DDD1C2; margin-bottom:.5rem;}
.brand {font-family:Georgia,"Times New Roman",serif; display:flex; align-items:center; gap:.65rem; color:#14253D; font-size:1rem; font-weight:650; line-height:1.2;}
.brand-icon {background:#14253D; color:#C5A47E; border-radius:10px; padding:.65rem; font-size:1.15rem;}
.brand small {font-family:ui-sans-serif,system-ui,sans-serif; display:block; font-size:.68rem; letter-spacing:.1em; font-weight:500; color:#615B60; margin-top:.3rem; text-transform:uppercase;}
.st-key-site_navigation [role="radiogroup"] {justify-content:flex-end; gap:.2rem; flex-wrap:wrap;}
.st-key-site_navigation [role="radiogroup"] label {color:#14253D; margin:0; padding:.55rem .75rem; border-radius:8px; transition:background .18s;}
.st-key-site_navigation [role="radiogroup"] label p {color:inherit;}
.st-key-site_navigation [role="radiogroup"] label:hover {background:#F5F0E9;}
.st-key-site_navigation [role="radiogroup"] label:has(input:checked) {background:#EEE5DF; color:#922F48;}
/* Hide only the decorative radio dot; keep the accessible native input. */
.st-key-site_navigation [data-testid="stRadioOption"] > div > div:first-child:not([data-testid="stMarkdownContainer"]) {display:none;}
.st-key-site_navigation [role="radiogroup"] label:focus-within {outline:2px solid #922F48; outline-offset:2px;}
.st-key-home_hero {background:#FFFFFF; border:1px solid #E5DDD3; border-radius:22px;
 padding:1.2rem; margin:.4rem 0; box-shadow:0 12px 36px #14253D0C;}
.st-key-home_hero .hero-photo img {height:clamp(300px,32vw,410px); object-position:center 25%;}
.st-key-home_actions {align-items:center; justify-content:center;}
.st-key-home_actions button {min-width:12rem;}
.st-key-home_hero .eyebrow {color:#922F48; text-transform:uppercase; letter-spacing:.13em;
 font-size:.7rem; font-weight:650; margin-bottom:.8rem; display:flex; align-items:center; gap:.5rem;}
.st-key-home_hero .eyebrow::before {content:''; width:20px; height:2px; background:#C5A47E;}
.st-key-home_hero h1.hero-title {font-family:Georgia,"Times New Roman",serif; text-align:center; font-size:clamp(1.8rem,3vw,2.5rem); font-weight:500; line-height:1.15;
 color:#14253D; letter-spacing:-.035em; margin:.25rem 0 .4rem;}
.st-key-home_hero .hero-title .hero-accent {color:#922F48;}
.st-key-home_hero .hero-copy {text-align:center; color:#514F55; font-size:.97rem; margin:.35rem 0 .6rem;}
.st-key-home_hero [data-testid="stCaptionContainer"] p {color:#615B60; font-size:.77rem;}
.hero-photo {position:relative; margin:0; border-radius:16px; overflow:hidden; background:#E5DDD3;}
.hero-photo::after {content:""; position:absolute; inset:0; background:rgba(20,37,61,.16); pointer-events:none;}
.hero-photo img {width:100%; height:390px; display:block; object-fit:cover; object-position:center 35%;}
.hero-photo figcaption {position:absolute; z-index:1; bottom:16px; left:16px; right:16px; background:#FFFFFFF2;
 border:1px solid #FFFFFF80; border-radius:10px; padding:.8rem 1rem; display:flex; justify-content:space-between;
 gap:.8rem; align-items:center; color:#14253D; font-size:.78rem; box-shadow:0 4px 16px #14253D15;}
.hero-photo figcaption strong {display:block; color:#14253D; font-size:.88rem;}
.hero-photo figcaption .photo-badge {background:#F5F0E9; color:#76233A; border-radius:20px;
 padding:.35rem .65rem; font-size:.7rem; white-space:nowrap;}
.editorial-intro {border-left:2px solid #C5A47E; padding:.2rem 0 .2rem 1rem; margin:0 0 1.2rem; max-width:48rem; color:#514F55;}
.editorial-intro strong {display:block; color:#14253D; font-family:Georgia,serif; font-size:1.3rem; font-weight:400; margin-bottom:.35rem;}
.asset-credit {text-align:right; font-size:.68rem; color:#615B60; margin:.45rem .2rem 0;}
.asset-credit a {color:#76233A; text-decoration:none;}
.booking-strip {display:flex; justify-content:space-between; align-items:center; gap:1rem;
 border-bottom:1px dashed #DDD1C2; padding:0 0 1rem; margin:0 0 .6rem; color:#14253D;}
.booking-strip .booking-title {font-size:1.05rem; font-weight:650;}
.booking-strip small {display:block; color:#615B60; font-size:.77rem; margin-top:.2rem;}
.booking-strip .ticket-label {background:#F5F0E9; color:#76233A; padding:.35rem .75rem; border-radius:20px; font-size:.73rem;}
.st-key-prediction_note [data-testid="stCaptionContainer"] p {color:#615B60;}
.st-key-prediction_results [data-testid="stMetric"] {border-top:3px solid #922F48; background:#FFFFFF;}
.st-key-prediction_results [data-testid="stColumn"]:last-child [data-testid="stMetric"] {border-top-color:#C5A47E; background:#F5F0E9;}
.st-key-analysis_kpis [data-testid="stColumn"]:nth-child(1) [data-testid="stMetric"] {border-top:3px solid #922F48; background:#FFFFFF;}
.st-key-analysis_kpis [data-testid="stColumn"]:nth-child(2) [data-testid="stMetric"] {border-top:3px solid #14253D; background:#FFFFFF;}
.st-key-analysis_kpis [data-testid="stColumn"]:nth-child(3) [data-testid="stMetric"] {border-top:3px solid #922F48; background:#FFFFFF;}
.st-key-analysis_kpis [data-testid="stColumn"]:nth-child(4) [data-testid="stMetric"] {border-top:3px solid #C5A47E; background:#F5F0E9;}
.feature-card {background:white; border:1px solid #E5DDD3; border-top:3px solid #14253D; border-radius:14px; padding:1.2rem; min-height:175px;}
.section-kicker {color:#922F48; font-size:.7rem; font-weight:600; letter-spacing:.13em; margin-top:1rem;}
.feature-card .number {font-size:.74rem; letter-spacing:.1em; color:#922F48;}
.feature-card.burgundy {border-top-color:#922F48;}
.feature-card.gold {border-top-color:#C5A47E;}
.feature-card h3 {font-size:1.06rem !important; margin:.5rem 0; padding:0;}
.feature-card p {font-size:.86rem; margin:0; color:#615B60;}
.site-footer {border-top:1px solid #DDD1C2; margin-top:1.6rem; padding-top:.85rem;
 color:#615B60; font-size:.75rem; display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap;}
@media(max-width:760px) {
 .block-container {padding:calc(3.75rem + .5rem) .85rem 1.5rem;}
 [data-testid="stHorizontalBlock"] {flex-direction:column; gap:.7rem;}
 [data-testid="stColumn"] {width:100% !important; flex:1 1 100% !important; min-width:0;}
 .st-key-site_navigation [data-testid="stHorizontalBlock"] {gap:1.4rem;}
 .st-key-site_navigation [role="radiogroup"] {justify-content:flex-start; gap:0;}
 .st-key-site_navigation [role="radiogroup"] label {padding:.5rem .55rem;}
 .st-key-site_navigation [role="radiogroup"] p {font-size:.8rem;}
 .st-key-home_hero {padding:.85rem;}
 .st-key-home_hero .hero-photo img {height:240px;}
 .st-key-home_actions [data-testid="stButton"], .st-key-home_actions button {width:100%;}
 .hero-photo img {height:270px;}
 .hero-photo .photo-badge {display:none;}
 .hero-photo figcaption {left:10px; right:10px; bottom:10px; padding:.65rem;}
 .booking-strip {gap:.5rem;}
 .booking-strip .ticket-label {white-space:nowrap;}
 .feature-card {min-height:0; padding:1rem;}
 [data-testid="stForm"] {padding:.9rem;}
 [data-testid="stMetricValue"] {font-size:1.35rem !important;}
}
@media(prefers-reduced-motion:reduce) {button,label {transition:none !important;}}
</style>''', unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(ROOT / 'models/flight_delay_rf_model_updated.pkl')


@st.cache_resource
def load_analysis():
    # Shared, read-only resource: never mutate this table in the UI.
    data = pd.read_parquet(DATA_PATH)
    metadata = json.loads(METADATA_PATH.read_text())
    if (int(data.flights.sum()) != metadata['totals']['flights']
            or int(data.delayed_flights.sum()) != metadata['totals']['delayed_flights']):
        raise ValueError('Analysis data does not match its metadata')
    return data, metadata


def prediction_dashboard():
    st.subheader('Flight Prediction')
    try:
        model = load_model()
    except (OSError, ValueError) as exc:
        st.error(f'The saved prediction model could not be loaded: {exc}')
        return
    encoder = model.named_steps['preprocessing'].named_transformers_['categorical'].named_steps['encoder']
    airline_codes, origin_codes, destination_codes = [sorted(c.tolist()) for c in encoder.categories_]
    airport_names = load_airport_names(ROOT / 'assets/airport_names.json')
    with st.form('flight_information'):
        departure, arrival = st.columns(2, gap='medium')
        with departure:
            origin = st.selectbox('Departure Airport', origin_codes, index=1,
                                  format_func=lambda code: airport_label(code, airport_names),
                                  help='Choose a valid direct route. Codes match the model’s 2024 airport categories.')
        with arrival:
            destination = st.selectbox('Destination Airport', destination_codes, index=2,
                                       format_func=lambda code: airport_label(code, airport_names))
        date_col, airline_col, time_col, distance_col = st.columns(4, gap='medium')
        with date_col:
            flight_date = st.date_input('Flight Date')
        with airline_col:
            airline = st.selectbox('Airline', airline_codes,
                                   index=airline_codes.index('AA') if 'AA' in airline_codes else 0)
        with time_col:
            flight_time = st.time_input('Scheduled Departure Time', value=time(12, 0),
                                        help='Local scheduled departure time.')
        with distance_col:
            distance = st.number_input('Flight Distance (miles)', min_value=1.0, value=500.0,
                                        help='Enter the distance of your direct route in miles.')
        submitted = st.form_submit_button('Predict Flight Delay', type='primary', icon=':material/flight_takeoff:', width='stretch')
    if submitted:
        if origin == destination:
            st.error('Departure and destination airports cannot be the same.')
        else:
            flight_data = pd.DataFrame([{
                'month': flight_date.month, 'day_of_month': flight_date.day,
                # Matches the training data: Monday=1, Sunday=7.
                'day_of_week': flight_date.weekday() + 1,
                'op_unique_carrier': airline.strip().upper(), 'origin': origin.strip().upper(),
                'dest': destination.strip().upper(),
                'crs_dep_time': flight_time.hour * 100 + flight_time.minute, 'distance': distance,
            }])
            probability = float(model.predict_proba(flight_data)[0][1])
            prediction = int(probability >= 0.50)
            risk = 'High' if probability >= 0.70 else 'Moderate' if probability >= 0.40 else 'Low'
            with st.container(key='prediction_results'):
                a, b, c = st.columns(3)
                a.metric('Predicted Status', 'Delayed' if prediction == 1 else 'On Time',
                         help='Delayed means departure is at least 15 minutes late. The classification threshold is 50%.')
                b.metric('Delay Probability', f'{probability:.1%}')
                c.metric('Risk Level', risk, help='Low <40%; Moderate 40–<70%; High ≥70%.')
    with st.container(key='prediction_note'):
        st.caption('Predictions are estimates based on 2024 historical data, not live flight status.')


def about_section():
    st.subheader('About Aviation Signature')
    st.write('Aviation Signature uses machine learning and historical flight data to estimate departure delay risk.')
    with st.container(border=True):
        st.markdown('#### How It Works')
        st.write('Enter your airports, date, airline, scheduled departure time, and distance. The Random Forest model estimates the probability of a departure delay of at least 15 minutes and returns a predicted status and risk level.')
    with st.container(border=True):
        st.markdown('#### Model Performance')
        st.caption('Evaluated on 106,105 flights from October–December 2024.')
        for col, label, value in zip(st.columns(4), ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                                      ['72.56%', '27.04%', '37.89%', '31.56%']):
            col.metric(label, value)
    with st.container(border=True):
        st.markdown('#### Data & Limitations')
        st.write('The system uses historical 2024 US flight data. Predictions are estimates, not live flight status or official flight information. Future flight patterns may differ from the training data, making estimates less reliable.')


def style_chart(chart, height):
    """Presentation only: retain the chart's data, encodings, and calculations."""
    # Streamlit 1.67 assigns padding.bottom in its frontend preprocessor.
    # Vega-Lite accepts a number here, but Streamlit requires an object.
    return (chart.properties(height=height, background='#FFFFFF',
                             padding={'left': 12, 'right': 12, 'top': 12, 'bottom': 20})
            .configure_view(stroke=None)
            .configure_axis(labelColor='#514F55', titleColor='#514F55', labelFontSize=11,
                            titleFontSize=11, titleFontWeight='normal', gridColor='#F0E9E1',
                            domainColor='#DDD1C2', tickColor='#DDD1C2', labelPadding=7,
                            titlePadding=10)
            .configure_axisX(grid=False, labelOverlap=True)
            .configure_legend(labelColor='#514F55', titleColor='#14253D', labelFontSize=11,
                              titleFontSize=11, titleFontWeight='normal'))


def rate_chart(table, x, title, order=None, line=False):
    st.markdown(f'#### {title}')
    if table.empty:
        st.info('No groups meet the minimum-flight threshold. Lower the threshold or broaden your filters.')
        return
    axis = alt.X(f'{x}:O', sort=order, title=None) if order is not None else alt.X(f'{x}:N', sort='-y', title=None)
    base = alt.Chart(table).encode(
        x=axis, y=alt.Y('delay_rate_pct:Q', title='Delayed departures (%)', scale=alt.Scale(zero=True)),
        tooltip=[alt.Tooltip(f'{x}:N', title=title), alt.Tooltip('flights:Q', format=',', title='Flights'),
                 alt.Tooltip('delayed_flights:Q', format=',', title='Delayed flights'),
                 alt.Tooltip('delay_rate_pct:Q', format='.2f', title='Delay rate (%)')])
    color = {'Month': BURGUNDY, 'departure_hour': NAVY, 'op_unique_carrier': BURGUNDY,
             'origin': '#6E5260', 'dest': '#9A784F', 'Weekday': NAVY}.get(x, NAVY)
    chart = (base.mark_line(color=color, strokeWidth=2, point=alt.OverlayMarkDef(size=28, filled=True))
             if line else base.mark_bar(color=color, cornerRadiusTopLeft=3, cornerRadiusTopRight=3))
    st.altair_chart(style_chart(chart, height=235), width='stretch')


def remember_filters():
    st.session_state['saved_analysis_filters'] = {
        key: value for key, value in st.session_state.items() if key.startswith('analysis_')}


def reset_filters():
    st.session_state.pop('saved_analysis_filters', None)
    for key in list(st.session_state):
        if key.startswith('analysis_'):
            del st.session_state[key]


def filter_select(label, options, key, formatter=str):
    values = st.multiselect(label, ['All'] + list(options),
                            default=st.session_state.get('saved_analysis_filters', {}).get(key, ['All']), key=key,
                            format_func=lambda value: 'All' if value == 'All' else formatter(value), on_change=remember_filters,
                            help='Remove All to select specific values. Clearing every value shows no matches.')
    return list(options) if 'All' in values else values


def analysis_dashboard():
    st.subheader('Flight Analytics')
    st.write('Explore flight delays by airline, airport, and month.')
    try:
        data, metadata = load_analysis()
    except (OSError, ValueError, KeyError) as exc:
        st.error(f'Analysis data could not be loaded: {exc}. Include both files from data/analysis in the application.')
        return
    st.caption(f"Historical {metadata['date_min'][:4]} US flights · Not live flight status.")
    # Render the filtered KPI cards above all collapsed controls and analyses.
    kpi_container = st.container(key='analysis_kpis')
    with st.container(border=True, key='analysis_filters'):
        st.markdown('#### Filters')
        left, middle, right = st.columns(3, gap='medium')
        with left:
            months = filter_select('Months', range(1, 13), 'analysis_months', lambda m: month_name[m])
            airlines = filter_select('Airlines', sorted(data.op_unique_carrier.unique()), 'analysis_airlines')
        with middle:
            origins = filter_select('Departure airports', sorted(data.origin.unique()), 'analysis_origins')
            destinations = filter_select('Destination airports', sorted(data.dest.unique()), 'analysis_destinations')
        with right:
            days = filter_select('Weekdays', range(1, 8), 'analysis_days', lambda d: WEEKDAYS[d - 1])
            hours = st.slider('Scheduled departure hours (local)', 0, 23, st.session_state.get('saved_analysis_filters', {}).get('analysis_hours', (0, 23)), key='analysis_hours',
                              on_change=remember_filters)
        threshold, reset = st.columns([3, 1], vertical_alignment='bottom')
        with threshold:
            minimum = st.number_input('Minimum flights per ranked group', min_value=1,
                                       value=st.session_state.get('saved_analysis_filters', {}).get('analysis_minimum', 100), step=10, key='analysis_minimum', on_change=remember_filters)
        with reset:
            st.button('Reset filters', on_click=reset_filters, width='stretch')
        st.caption('Clear a selection to show no matches. Minimum flights affects rankings, heatmap cells, and insights; trend charts retain all available groups.')
    selected = filter_data(data, {'month': months, 'op_unique_carrier': airlines,
                                'origin': origins, 'dest': destinations, 'day_of_week': days}, hours)
    values = totals(selected)
    if not values['flights']:
        st.warning('No flights match these filters. Select at least one value in every filter, broaden your route or time selection, or reset filters.')
        return
    with kpi_container:
        a, b, c, d = st.columns(4)
        a.metric('Flights analyzed', f"{values['flights']:,}")
        b.metric('Delayed departures', f"{values['delayed_flights']:,}")
        c.metric('Departure delay rate', f"{values['delay_rate']:.2%}")
        d.metric('Average departure delay', f"{values['average_departure_delay']:.2f} min")
    delayed_average = values['average_delayed_departure']
    monthly = summarize(selected, ['month'])
    monthly['Month'] = monthly.month.map(lambda m: month_abbr[m])
    hourly = summarize(selected, ['departure_hour'])
    daily = summarize(selected, ['day_of_week'])
    daily['Weekday'] = daily.day_of_week.map(lambda d: WEEKDAYS[d - 1])
    carrier = summarize(selected, ['op_unique_carrier'])
    origin = summarize(selected, ['origin'])
    destination = summarize(selected, ['dest'])
    with st.expander('Monthly & Time Analysis', expanded=False):
        left, right = st.columns(2, gap='medium')
        with left:
            rate_chart(monthly, 'Month', 'Monthly delay rate', list(month_abbr)[1:], line=True)
        with right:
            rate_chart(hourly, 'departure_hour', 'Scheduled departure hour', list(range(24)), line=True)
        left, right = st.columns(2, gap='medium')
        with left:
            rate_chart(daily, 'Weekday', 'Weekday delay rate', WEEKDAYS)
        with right:
            st.markdown('#### Weekday–hour heatmap')
            heat = summarize(selected, ['day_of_week', 'departure_hour'])
            heat = heat[heat.flights >= minimum].copy()
            heat['Weekday'] = heat.day_of_week.map(lambda d: WEEKDAYS[d - 1])
            if heat.empty:
                st.info('No heatmap cells meet the minimum-flight threshold.')
            else:
                chart = alt.Chart(heat).mark_rect(stroke='#FFFFFF', strokeWidth=.7, cornerRadius=2).encode(
                    x=alt.X('departure_hour:O', title='Scheduled departure hour (local)', sort=list(range(24)),
                            axis=alt.Axis(labelAngle=0, labelExpr="datum.value % 3 == 0 ? datum.label : ''")),
                    y=alt.Y('Weekday:N', sort=WEEKDAYS, title=None),
                    color=alt.Color('delay_rate_pct:Q', title='Delay rate (%)',
                                    scale=alt.Scale(range=['#F5F0E9', '#D8C5B1', '#B38087', '#922F48', '#14253D'], domainMin=0),
                                    legend=alt.Legend(orient='bottom', direction='horizontal', gradientLength=180,
                                                      gradientThickness=10, format='.0f', titlePadding=8)),
                    tooltip=['Weekday:N', alt.Tooltip('departure_hour:O', title='Hour'),
                             alt.Tooltip('flights:Q', format=','), alt.Tooltip('delayed_flights:Q', format=','),
                             alt.Tooltip('delay_rate_pct:Q', format='.2f')])
                st.altair_chart(style_chart(chart, height=245), width='stretch')
            st.caption(f'Rankings and heatmap require at least {minimum:,} flights per displayed group. Missing cells are unavailable or below the threshold; they do not mean zero delay.')

    with st.expander('Airline Performance', expanded=False):
        rate_chart(carrier[carrier.flights >= minimum], 'op_unique_carrier', 'Airline delay rate')
    with st.expander('Airport Performance', expanded=False):
        left, right = st.columns(2, gap='medium')
        with left:
            rate_chart(origin[origin.flights >= minimum].nlargest(15, 'delay_rate_pct'), 'origin', 'Departure airport ranking · top 15')
        with right:
            rate_chart(destination[destination.flights >= minimum].nlargest(15, 'delay_rate_pct'), 'dest', 'Destination airport ranking · top 15')
    with st.expander('Detailed Data', expanded=False):
        st.markdown('#### Insights')
        st.caption('Average departure delay includes negative values for early departures. ' +
                   (f'Among delayed flights only: {delayed_average:.2f} minutes.' if delayed_average is not None else 'No delayed flights in this selection.'))
        eligible_months = monthly[monthly.flights >= minimum]
        eligible_carriers = carrier[carrier.flights >= minimum]
        if len(eligible_months) >= 2:
            high = eligible_months.loc[eligible_months.delay_rate_pct.idxmax()]
            low = eligible_months.loc[eligible_months.delay_rate_pct.idxmin()]
            st.write(f"• Highest observed monthly rate: **{high['Month']} ({high.delay_rate_pct:.2f}%)**; lowest: **{low['Month']} ({low.delay_rate_pct:.2f}%)**. "
                     f'The difference is {high.delay_rate_pct - low.delay_rate_pct:.2f} percentage points.')
        else:
            st.write('Select at least two months with enough flights for a monthly comparison.')
        if len(eligible_carriers) >= 2:
            row = eligible_carriers.loc[eligible_carriers.delay_rate_pct.idxmax()]
            st.write(f'• Highest observed airline rate: **{row.op_unique_carrier} ({row.delay_rate_pct:.2f}%)**, across **{int(row.flights):,} flights**.')
        st.caption('Rates describe historical associations, not causes or guaranteed future performance. Airlines serve different routes and schedules; ties may occur in rankings.')
        st.divider()
        st.markdown('#### Summary table')
        grouping = st.selectbox('Summarize by', ['month', 'op_unique_carrier', 'origin', 'dest', 'day_of_week', 'departure_hour'])
        detail = summarize(selected, [grouping])
        st.dataframe(detail, hide_index=True, width='stretch')
        st.download_button('Download filtered summary (CSV)', detail.to_csv(index=False).encode(),
                           file_name=f'flight_delay_2024_{grouping}.csv', mime='text/csv', on_click='ignore')
        st.divider()
        st.markdown('#### Data notes')
        st.caption(f"Source coverage: {metadata['date_min']} to {metadata['date_max']} · "
                   f"{metadata['totals']['flights']:,} eligible flights · Cancelled flights excluded; diverted flights retained.")
        st.write('Full cleaned 2024 data, not a sample. A delayed departure is at least 15 minutes late. Rates divide summed delayed counts by summed flight counts, never average percentages.')
        st.write('Weekdays: Monday=1 through Sunday=7. Scheduled 2400 is grouped into hour 00 on its recorded date. Cancelled flights and records with missing departure delay are excluded.')
        st.caption('The compact data supports month-level filtering, not individual flight records, exact dates, medians, or delay distributions. Analysis airport coverage is independent of the model’s training categories.')
        st.download_button('Download data metadata (JSON)', json.dumps(metadata, indent=2),
                           file_name=METADATA_PATH.name, mime='application/json', on_click='ignore')



def navigate(section):
    st.session_state['site_section'] = section


@st.cache_data
def hero_image_uri():
    # Self-contained image: no external request or expiring CDN link at runtime.
    return 'data:image/jpeg;base64,' + base64.b64encode(HERO_PATH.read_bytes()).decode('ascii')


def home_section():
    with st.container(key='home_hero'):
        st.html(f'''<figure class="hero-photo"><img src="{hero_image_uri()}" alt="Passenger jet in flight against blue sky and sunlit clouds" width="1920" height="1440" fetchpriority="high"></figure><div class="asset-credit">Photo: <a href="https://www.pexels.com/photo/white-and-red-plane-beside-clouds-low-angle-photography-46148/" target="_blank" rel="noopener noreferrer">Pixabay / Pexels</a></div>''')
        st.html('<h1 class="hero-title">Fly Smarter. <span class="hero-accent">Travel with Confidence.</span></h1><p class="hero-copy">Predict flight delays and explore travel trends.</p>')
        with st.container(key='home_actions', horizontal=True, horizontal_alignment='center', gap='small'):
            st.button('Predict Flight', type='primary', on_click=navigate,
                      args=('Flight Prediction',), icon=':material/flight_takeoff:')
            st.button('View Analytics', on_click=navigate,
                      args=('Flight Analytics',), icon=':material/monitoring:')


with st.container(key='site_navigation'):
    brand, navigation = st.columns([1, 2], vertical_alignment='center')
    with brand:
        st.markdown('<div class="brand"><span class="brand-icon" aria-hidden="true">✈</span><span>AVIATION SIGNATURE<small>Flight intelligence</small></span></div>', unsafe_allow_html=True)
    with navigation:
        section = st.radio('Website navigation', ['Home', 'Flight Prediction', 'Flight Analytics', 'About'],
                           horizontal=True, key='site_section', label_visibility='collapsed')
if section == 'Home':
    home_section()
elif section == 'Flight Prediction':
    prediction_dashboard()
elif section == 'Flight Analytics':
    analysis_dashboard()
else:
    about_section()
st.markdown('<footer class="site-footer"><span>AVIATION SIGNATURE</span></footer>', unsafe_allow_html=True)
