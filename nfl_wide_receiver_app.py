import pandas as pd 
import numpy as np
import streamlit as st
import base64
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import plotly.express as px

# python -m streamlit run nfl_wide_receiver_app.py

st.header("NFL Receiving Stats Explorer")

st.markdown("This app performs webscraping of NFL receiving stats from https://www.pro-football-reference.com/ & provides visualizations to go along with the Wide Receiver Grading tool.")

st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1978,2026))))

# Webscraping of NFL player stats
# https://www.pro-football-reference.com/years/2025/passing.htm
@st.cache_data

def load_data(year):
    url = "https://www.pro-football-reference.com/years/" + str(year) + "/receiving.htm"
    html = pd.read_html(url, header=1)
    df = html[0]
    raw = df.drop(df[df.Age == 'Age'].index) # Deletes repeating headers
    raw = raw.fillna(0)
    raw = raw.drop_duplicates(['Player'], keep='first')
    playerstats = raw.drop(['Rk'], axis=1)
    return playerstats
playerstats = load_data(selected_year)

# Sidebar - Team selection
# Clean and sort team names, handling potential NaN values

try:
  unique_teams = playerstats['Team'].dropna().unique()
  sorted_uni_team = sorted([str(team) for team in unique_teams])
except Exception as e:
  st.error(f"Error processing team data: {e}")
  sorted_uni_team = []
    
    
selected_team = st.sidebar.multiselect('Team', sorted_uni_team, sorted_uni_team)

# Sidebar - Position selection

unique_pos = ['WR', 'TE', 'RB', 'FB']
selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)

df_selected_team = playerstats[(playerstats.Team.isin(selected_team)) & (playerstats.Pos.isin(selected_pos))]
st.dataframe(df_selected_team)

# Download NFL player stats data

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64",{b64} download="playerstats.csv">Download CSV File</a>'
    return href

# st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)

#Heatmap
if st.button('Intercorrelation Heatmap'):
    st.header('Intercorrelation Matrix Heatmap')
    df_selected_team.to_csv('output.csv', index=False)
    df = pd.read_csv('output.csv')

    corr = df.drop(columns=['Player', 'Team', 'Pos', 'Awards']).corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True, cmap='BuPu')
    st.pyplot(f)

# Scatter Plot
st.sidebar.header('Scatter Plot Settings')
x_axis = st.sidebar.selectbox('X Axis', ['G', 'Tgt', 'Rec', 'Yds', 'Y/R', 'TD', '1D', 'Succ%', 'R/G', 'Y/G', 'Ctch%', 'Y/Tgt', 'Fmb'])
y_axis = st.sidebar.selectbox('Y Axis', ['G', 'Tgt', 'Rec', 'Yds', 'Y/R', 'TD', '1D', 'Succ%', 'R/G', 'Y/G', 'Ctch%', 'Y/Tgt', 'Fmb'])

if st.sidebar.button('Generate Scatter Plot'):
    st.header('Scatter Plot between ' + x_axis + ' and ' + y_axis)
    
    # Scatter plot with OLS (Ordinary Least Sqaures) trendline 

    fig = px.scatter(df_selected_team, x=x_axis, y=y_axis, hover_data=['Player'], trendline='ols')
    st.plotly_chart(fig)


# Average Yards per position

# Get average yards by position

pos_avg = df_selected_team.groupby('Pos')['Y/G'].mean().reset_index().round(2)
pos_avg = pos_avg.sort_values(by='Y/G', ascending=False)

if st.button('Average YPG by Position'):
    st.header('Average Yards Per Game by Position for the Year of ' + str(selected_year))
 
    fig = px.bar(pos_avg, x='Pos', y='Y/G', text='Y/G')
    st.plotly_chart(fig)
    
# Find Best Graded Receivers

st.header('Top Graded Receivers in ' + str(selected_year))

st.markdown(" A player's Receiving Grade is calculated by normalizing all significant receiving statistics and then adding them together. The number that is derived from the previous step is then scaled on a range from 0 to 100 (100 being the highest possible score) with a penalty assigned for a player's total fumbles and the percentage of targets they did not catch.")

dfs_copy = df_selected_team.copy()
dfs_copy = dfs_copy.drop(columns=['Age'])
stats = dfs_copy.select_dtypes(include=[np.number])

# Normalizing stats using MinMaxScaler

try:
    normalizer = MinMaxScaler()
    normalized_stats = normalizer.fit_transform(stats)
    normalized_df = pd.DataFrame(normalized_stats, columns=stats.columns, index=dfs_copy.index)
# Calculate Receiving Grade 
    normalized_df['Receiving Grade'] = normalized_df.sum(numeric_only=True, axis=1)
    normalized_df['Receiving Grade'] = (normalized_df['Receiving Grade'] - normalized_df['Receiving Grade'].min()) / (normalized_df['Receiving Grade'].max() - normalized_df['Receiving Grade'].min()) * 100 - normalized_df['Fmb'] - 1 - normalized_df['Ctch%']
# Adding Rank Column for readability
    rank_df = pd.concat([df_selected_team, normalized_df['Receiving Grade']], axis=1)
    rank_df['Rank'] = rank_df['Receiving Grade'].rank(ascending=False)
    st.write(rank_df[['Rank','Player', 'Age', 'Team', 'Pos', 'Rec', 'Yds', 'Receiving Grade']].sort_values(by='Receiving Grade', ascending=False).head().round(2))
except ValueError:
    pass
# Adding Individual player grade lookup

st.write("Player Grade Lookup:")

player_choice = st.selectbox('Player', df_selected_team['Player'].unique())

if st.button('Show Player Grade'):
    player_data = rank_df[rank_df['Player'] == player_choice]
    player = player_data[['Rank', 'Player', 'Age', 'Team', 'Pos', 'Rec', 'Yds', 'Receiving Grade']].round(2)

    st.write(player)





































































