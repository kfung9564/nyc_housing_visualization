import pandas as pd
import json
from area import area
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv('Neighborhood_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_mon.csv')
df = df[df['City'] == 'New York']

nycmap = json.load(open("nyc_neighborhoods.geojson"))

dates = ('2019-12-31',
         '2020-01-31',
         '2020-02-29',
         '2020-03-31',
         '2020-04-30',
         '2020-05-31',
         '2020-06-30',
         '2020-07-31',
         '2020-08-31',
         '2020-09-30',
         '2020-10-31')

new_frame = pd.read_csv('neighborhood_prices_modified.csv').drop(columns=['Unnamed: 0'])
income_loc = pd.read_csv('neighborhood_locations_income.csv')
income_df = pd.read_csv('median_incomes_nyc_2019.csv')

income_df.rename(columns={'Data': 'median_income'})
for i in range(len(income_loc)):
    income_loc.loc[i, 'median_income'] = \
        income_df.loc[income_df['Location'] == income_loc.loc[i, 'income_area'], 'Data'].values[0]

# create dictionary of nta codes mapping to area (square miles)
d = {}

neighborhood = nycmap["features"]
for n in neighborhood:
    name = n["properties"]["ntaname"]
    a = area(n["geometry"]) / (1609 * 1609)  # converts from m^2 to mi^2
    d[name] = a

# create new columns in df for area
new_frame["area"] = new_frame["ntaname"].map(d)
new_frame = new_frame.dropna(subset=["area"])

# add income to df
new_frame.insert(2, 'median_income', income_loc['median_income'])

# new_frame = new_frame.loc[new_frame['count'] > 0]

dates_frame = pd.DataFrame()
temp = new_frame

for r in range(len(temp)):
    date_row = temp.iloc[r:r + 1, :]
    curr_region = date_row.iloc[0, 0]
    curr_boro = date_row.iloc[0, 1]
    curr_income = date_row.iloc[0, 2]
    curr_area = date_row.iloc[0, len(date_row.columns) - 1]
    dates_transposed = date_row.iloc[:, len(date_row.columns) - len(dates) - 2:len(date_row.columns) - 1].T
    dates_transposed.reset_index(inplace=True)
    dates_transposed.columns = ['date', 'price']
    dates_transposed['ntaname'] = curr_region
    dates_transposed['boro_name'] = curr_boro
    dates_transposed['median_income'] = curr_income
    dates_transposed['area'] = curr_area
    dates_frame = pd.concat([dates_frame, dates_transposed])

dates_frame.reset_index(inplace=True)

app.layout = html.Div([
    html.Div([
        dcc.Dropdown(
            id='map-dropdown',
            options=[
                {'label': 'Typical Home Value', 'value': 'price'},
                {'label': 'Typical Home Value Changes ($)', 'value': 'price_change'},
                {'label': 'Typical Home Value Changes (%)', 'value': 'price_change_per'},
                {'label': 'Median Household Income', 'value': 'median_income'}
            ],
            value='price',
            clearable=False
        ),
        dcc.Graph(id='graph-with-slider'),
        dcc.Slider(
            id='date-slider',
            min=1,
            max=len(dates) - 1,
            value=1,
            step=1,
            dots=True,
            updatemode='mouseup'
        ),
        html.Div(id='updatemode-output-container'),
        # html.Div([html.Pre(id='selected-data')]),
        html.Label(['X: ',
                    dcc.Dropdown(
                        id='dot-dropdown-x',
                        options=[
                            {'label': 'Typical Home Value', 'value': 'price'},
                            {'label': 'Typical Home Value Changes ($)', 'value': 'price_change'},
                            {'label': 'Typical Home Value Changes (%)', 'value': 'price_change_per'},
                            {'label': 'Median Household Income', 'value': 'median_income'},
                            {'label': 'Neighborhood Size', 'value': 'area'}
                        ],
                        value='price',
                        clearable=False
                    )
                    ], className='dot-x-label'),
        html.Label(['Y: ',
                    dcc.Dropdown(
                        id='dot-dropdown-y',
                        options=[
                            {'label': 'Typical Home Value', 'value': 'price'},
                            {'label': 'Typical Home Value Changes ($)', 'value': 'price_change'},
                            {'label': 'Typical Home Value Changes (%)', 'value': 'price_change_per'},
                            {'label': 'Median Household Income', 'value': 'median_income'},
                            {'label': 'Neighborhood Size', 'value': 'area'}
                        ],
                        value='price_change_per',
                        clearable=False
                    )
                    ], className='dot-y-label'),
        html.Label(['Size: ',
                    dcc.Dropdown(
                        id='dot-dropdown-size',
                        options=[
                            {'label': 'Typical Home Value', 'value': 'price'},
                            {'label': 'Median Household Income', 'value': 'median_income'},
                            {'label': 'Neighborhood Size', 'value': 'area'},
                        ],
                        value='median_income',
                        clearable=False
                    )
                    ], className='dot-size-label'),
    ], className='map-div'),
    html.Div([
        dcc.Dropdown(
            id='scatter-dropdown',
            options=[
                {'label': 'New York City', 'value': 'New York City'},
                {'label': 'Bronx', 'value': 'Bronx'},
                {'label': 'Brooklyn', 'value': 'Brooklyn'},
                {'label': 'Manhattan', 'value': 'Manhattan'},
                {'label': 'Queens', 'value': 'Queens'},
                {'label': 'Staten Island', 'value': 'Staten Island'}
            ],
            clearable=True,
            multi=True
        ),
        dcc.Graph(id='scatter-graph'),
        html.Div([
            dcc.Graph(id='dot-graph')
        ], className='dot-div')
    ], className='scatter-div'),
], className='whole', style={'display': 'flex'})


@app.callback(
    Output('graph-with-slider', 'figure'),
    [Input('date-slider', 'value')],
    [Input('map-dropdown', 'value')])
def update_figure(selected_date, selected_view):
    selected = dates[selected_date]
    selected_df = None
    map_title = ''

    if selected_view == 'price_change':
        map_title = 'New York City Typical Home Values Changes ($)'

        start_prices = dates_frame.loc[dates_frame['date'] == dates[0]]
        filtered_df = dates_frame.loc[dates_frame['date'] == selected]
        filtered_df['price_change'] = filtered_df['price'].values - start_prices['price'].values
        selected_df = filtered_df
        selected = selected_view

    elif selected_view == 'price_change_per':
        map_title = 'New York City Typical Home Values Changes (%)'

        start_prices = dates_frame.loc[(dates_frame['date'] == dates[0])]
        filtered_df = dates_frame.loc[dates_frame['date'] == selected]
        with np.errstate(divide='ignore', invalid='ignore'):
            filtered_df['price_change_per'] = ((filtered_df['price'].values - start_prices['price'].values) /
                                               start_prices['price'].values) * 100

        filtered_df.loc[np.isnan(filtered_df['price_change_per']), 'price_change_per'] = 0
        selected_df = filtered_df
        selected = selected_view

    elif selected_view == 'price':
        map_title = 'New York City Typical Home Values'

        selected = dates[selected_date]
        selected_df = new_frame

    elif selected_view == 'median_income':
        map_title = 'New York City Median Household Income'

        selected = 'median_income'
        selected_df = income_loc

    fig = px.choropleth_mapbox(selected_df,
                               geojson=nycmap,
                               locations="ntaname",
                               custom_data=["ntaname", "boro_name"],
                               featureidkey="properties.ntaname",
                               color=selected,
                               color_continuous_scale="aggrnyl",
                               mapbox_style="carto-positron",
                               zoom=9.7, center={"lat": 40.7, "lon": -73.9},
                               opacity=0.7,
                               hover_name="ntaname",
                               labels={'ntaname': 'NeighborHood',
                                       'boro_name': 'Borough',
                                       selected: selected_view,
                                       },
                               title=map_title
                               )

    fig.update_layout(clickmode='event+select',
                      transition_duration=500,
                      autosize=False,
                      )

    return fig


@app.callback(
    Output('updatemode-output-container', 'children'),
    [Input('date-slider', 'value')])
def display_value(value):
    return "Date: " + dates[value]


@app.callback(
    Output('scatter-graph', 'figure'),
    [Input('graph-with-slider', 'selectedData')],
    [Input('scatter-dropdown', 'value')])
def update_scatter(selected, selected_boro):
    if selected is None:
        # PreventUpdate prevents ALL outputs updating
        raise dash.exceptions.PreventUpdate

    selected_frame = pd.DataFrame()
    price_per_frame = new_frame.copy()
    start_prices = price_per_frame[dates[0]]
    for date in dates:
        if date == dates[0]:
            continue

        with np.errstate(divide='ignore', invalid='ignore'):
            price_per_frame[date] = ((price_per_frame[date].values - start_prices.values) /
                                     start_prices.values) * 100

        price_per_frame.loc[np.isnan(price_per_frame[date]), date] = 0

    price_per_frame = price_per_frame.drop(columns=[dates[0]])

    if selected_boro is not None:
        for borough in selected_boro:
            current_boro = price_per_frame.loc[price_per_frame['count'] > 0]
            print(borough)

            if borough != 'New York City':
                current_boro = price_per_frame.loc[
                    (price_per_frame['boro_name'] == borough) & (price_per_frame['count'] > 0)]

            current_boro.loc[borough] = current_boro.mean()
            current_boro.iloc[-1, 0] = borough
            current_boro.iloc[-1, 1] = borough
            price_per_frame = price_per_frame.append(current_boro.iloc[-1, :], ignore_index=True)
            selected['points'].append({'curveNumber': 0, 'pointNumber': 0, 'pointIndex': 0,
                                       'location': borough, 'z': 0,
                                       'hovertext': borough,
                                       'customdata': [borough, borough]})

    for s in selected['points']:
        selected_region = s['customdata'][0]
        selected_boro = s['customdata'][1]
        sf = price_per_frame.loc[
            (price_per_frame['ntaname'] == selected_region) & (price_per_frame['boro_name'] == selected_boro)].drop(
            columns=["area"])

        transposed = sf.iloc[:, len(sf.columns) - len(dates):].T
        transposed.reset_index(inplace=True)
        transposed.columns = ['date', 'price_per']
        transposed['ntaname'] = selected_region
        transposed['boro_name'] = selected_boro
        selected_frame = pd.concat([selected_frame, transposed])

    fig = px.line(selected_frame,
                  x='date',
                  y='price_per',
                  color='ntaname',
                  labels={
                      'date': 'Date',
                      'price_per': 'Change in Price (%)',
                      'ntaname': 'NeighborHood',
                  },
                  title='Change In Home Value On Selected Areas (%)'
                  )

    fig.update_yaxes(ticksuffix="%")
    fig.update_traces(mode="markers+lines", hovertemplate=None)
    fig.update_layout(hovermode="x unified")

    return fig


@app.callback(
    Output('dot-graph', 'figure'),
    [Input('date-slider', 'value')],
    [Input('dot-dropdown-x', 'value')],
    [Input('dot-dropdown-y', 'value')],
    [Input('dot-dropdown-size', 'value')])
def update_dot_graph(selected_date, selected_x, selected_y, selected_size):
    selected = dates[selected_date]
    start_prices = dates_frame.loc[dates_frame['date'] == dates[0]]
    filtered_df = dates_frame.loc[dates_frame['date'] == selected]

    filtered_df['price_change'] = filtered_df['price'].values - start_prices['price'].values

    with np.errstate(divide='ignore', invalid='ignore'):
        filtered_df['price_change_per'] = ((filtered_df['price'].values - start_prices['price'].values) / start_prices[
            'price'].values) * 100

    filtered_df.loc[np.isnan(filtered_df['price_change_per']), 'price_change_per'] = 0

    filtered_df = filtered_df.loc[filtered_df['price_change'] != 0]

    chart_title = selected_x + ' by ' + selected_y + ' by ' + selected_size + ' on ' + str(selected)

    fig = px.scatter(filtered_df, x=selected_x, y=selected_y,
                     size=selected_size, color='boro_name', hover_name='ntaname',
                     log_x=False, size_max=30,
                     labels={
                         'price': 'Value ($)',
                         'price_change': 'Change in Value ($)',
                         'price_change_per': 'Change in Value (%)',
                         'median_income': 'Median Income ($)',
                         'area': 'Area',
                         'boro_name': 'Borough',
                         'ntaname': 'Neighborhood'
                     },
                     title=chart_title
                     )

    fig.update_layout(transition_duration=0)

    return fig


# @app.callback(Output('selected-data', 'children'),
#               [Input('graph-with-slider', 'selectedData')])
# def dis_play_hover_data(selected_data):
#    return json.dumps(selected_data, indent=2)


if __name__ == '__main__':
    app.run_server(debug=True)
