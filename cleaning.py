#Kian Ankerson
#CptS 315
#Description:
#This python script reads in my data files, which are separated into by year into a per game and advance dataset.
#In this script I perform nearly all of the cleaning to the data before it is used to train models
#I first read in my files, then rows of data where a player had switched teams to only include their total statistics
#I then drop extra columns from the data that do not provide a statistical significance including the Rk and Team columns
#I had to clean up the player names, as they were given in an name/id format with special chcaracters
#I then adjusted the position of players, if they had multiple positions, there first listed position was their primary
#I then grouped the players into frontcourt and backcourt groups as to how players are voted for.
#I also set the year column to match my all_star data, detailing who were the all stars for a seasons
#I decided to set some minimum requirements to exclude big outliers in my dataset
#this is particularly helpful for stats that get expanded to per 48 minutes
#after dropping the outliers, I made sure I didn't exlclude any of the all stars
#After joining my per game and advanced dataframes, I used my all star data to label my statistics dataframe
#I then finalized 3 different datasets.
#My first data set consisted of only -1 and 1 representing whether or not a player was above or below their average for their
#position and year.
#My next dataset consisted of the variances, the stats minus their groups average
#My last dataset consisted of the z score for the groups
#I also save what statistics are in which columns for help later on.


import pandas as pd
import unidecode
import numpy as np

years = ['2002-03', '2003-04', '2004-05', '2005-06', '2006-07', '2007-08', '2008-09', '2009-10', '2010-11', '2011-12',
         '2012-13', '2013-14', '2014-15', '2015-16', '2016-17', '2017-18', '2018-19', '2019-20', '2020-21', '2021-22']

def read_per_game():
    base = "_per_game.csv"


    list_of_df = []
    #read all files
    for year in years:
        #set file name
        filename = year+base
        #read file
        temp_df = pd.read_csv(filename)

        #if a player appears multiple times, they switched teams so we need to keep only the column with the total
        #first sort by player and g. the Total column should be the one appearing first with it sorted
        temp_df.sort_values(by=['Player', 'G'], ascending= [True, False], inplace=True, ignore_index=True)
        temp_df.drop_duplicates(['Player'], inplace=True)

        #add a year column
        temp_df['year'] = year
        list_of_df.append(temp_df)

    df = pd.concat(list_of_df)
    return df


def read_adv():
    base = "_advanced.csv"

    list_of_df = []
    # read all files
    for year in years:
        # set file name
        filename = year + base
        # read file
        temp_df = pd.read_csv(filename)

        # if a player appears multiple times, they switched teams so we need to keep only the column with the total
        # first sort by player and g. the Total column should be the one appearing first with it sorted
        temp_df.sort_values(by=['Player', 'G'], ascending=[True, False], inplace=True, ignore_index=True)
        temp_df.drop_duplicates(['Player'], inplace=True)
        temp_df['year'] = year
        list_of_df.append(temp_df)

    df = pd.concat(list_of_df)
    return df

def clean_both(df):
    #df = pd.DataFrame()
    #Drop unneeded columns
    df.drop(columns=['Rk', 'Tm'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    #split the player column into their id and name
    df[['player_name','player_id']] = pd.DataFrame(df.Player.str.split('\\').tolist(), columns=['player_name', 'player_id'])
    #remove asterisk, players with asterisk are marked becase they are in the hof
    df['player_name'] = df.player_name.str.replace('*', '')
    #turn accented string into normal
    df['player_name'] = df['player_name'] = df.apply(lambda row: unidecode.unidecode(row.player_name), axis=1)
    df.drop(columns=['Player'], inplace=True)
    return df

def clean_position(string):
    if '-' in string:
        string = string.split('-')
        string = string[0]
    return string

def guard_or_big(string):
    if string == 'PG' or string == 'SG':
        return 'backcourt'
    else:
        return 'frontcourt'

def set_year(string):
    return int("20" + str(string)[-2:])

def check_in_data(player_year_g_df):
    all_stars = pd.read_csv('all_stars_clean.csv')
    all_stars.rename(columns={'player':"player_name", 'year':"Year"}, inplace=True)

    #left join
    left_join = pd.merge(all_stars, player_year_g_df, how='left', on=['Year', 'player_name'])
    #check if any rows where null
    is_nan = left_join[left_join.isnull().any(axis=1)]
    if is_nan.shape[0] > 0:
        print("Found all star not in your stats!")
        print(is_nan)
        raise KeyError()
    print("All players found! All_stars: ", str(all_stars.shape[0]), ",  rows: ", str(player_year_g_df.shape[0]))
    return

def is_all_star(row_player, row_year, all_star_df):
    try:
        #if this works, then they are an all star that year
        all_star_df[(all_star_df['year'] == row_year) & (all_star_df[row_player])]
        return 1
    except KeyError:
        #not found not an all_star
        return -1




def main():
    pg = read_per_game()
    pg = clean_both(pg)
    #rename the MP column for joining
    pg.rename(columns={'MP': "MPG"}, inplace=True)

    adv = read_adv()
    adv = clean_both(adv)
    adv.drop(['Unnamed: 19', 'Unnamed: 24'], inplace=True, axis=1)

    all_stars = pd.read_csv('all_stars_clean.csv')
    all_stars['is_all_star'] = 1
    all_stars.rename(columns={'player': "player_name", 'year': "Year"}, inplace=True)
    #Join the data

    all_data = pd.merge(pg, adv)



    #Adjust positions. NBA all stars select 2 guards, and 3 front court players, guard = PG,SG, front court = SF,PF,C
    #calculate averages for each position by year;
    #If a player plays in multiple positions, their primary position is listed first we want that one
    all_data['true_pos'] = all_data.apply(lambda row: clean_position(row.Pos), axis=1)
    all_data['pos'] = all_data.apply(lambda row: guard_or_big(row.true_pos), axis=1)
    #set the years to the single numeric year
    all_data['Year'] = all_data.apply(lambda row: set_year(row.year)  , axis=1)





    all_data.drop(columns=['true_pos','Pos', 'year'], inplace=True)

    #I want to drop rows with big outliers, that create unrealistic stats when estimated per game, or per 48 mins
    #
    fga_min = 1
    minutes_per_game_min = 15
    games_min = 4
    total_minutes_min = 50
    all_data = all_data[(all_data['G'] >= games_min) & (all_data['MP'] >= total_minutes_min) & (all_data['MPG'] >= minutes_per_game_min) & (all_data['FGA'] >= fga_min)]
    #check that I have stats for all of the all stars
    check_in_data(all_data[['player_name', 'Year', 'G']])

    #averages for each pos and year
    averages = all_data.groupby(['pos', 'Year']).mean()

    #df with only names, pos, year
    pos_name_year = all_data[['pos', 'player_name', 'Year']]

    #normalize the data to only signs
    normalized = all_data.groupby(['pos', 'Year']).transform(lambda x: (x-x.mean()))
    #get the sign
    normalized = np.sign(normalized)
    normalized = normalized.join(pos_name_year,how='outer')
    # label the all stars as 1 using left join
    normalized = pd.merge(normalized, all_stars, how='left', on=['Year', 'player_name'])
    # check that I didn't miss any
    print(normalized['is_all_star'].sum(skipna=True))
    print(all_stars.shape[0])
    # set the nan to -1 for all stats, if it was nan, they they didn't meet a requirement such as shooting enough shots
    normalized.fillna(value=-1, inplace=True)

    #if we only have -1, or +1, representing whether or not they were higher than the average, then we might be okay
    #with not adjusting by trends over years, like faster pace

    #the input to our predictor model, would first be normalized then
    normalized.drop(columns=['Year', 'pos', 'player_name'], inplace=True)
    #save as csv
    normalized.to_csv('data_normalized.csv', index=False)



    #What if we just enter the difference between their stat and the mean?
    normalized = all_data.groupby(['pos', 'Year']).transform(lambda x: (x - x.mean()))
    normalized = normalized.join(pos_name_year, how='outer')
    # label the all stars as 1 using left join
    normalized = pd.merge(normalized, all_stars, how='left', on=['Year', 'player_name'])
    # check that I didn't miss any
    print(normalized['is_all_star'].sum(skipna=True))
    print(all_stars.shape[0])
    # set the nan to 0 for all stats,
    normalized.fillna(value=0, inplace=True)
    normalized.drop(columns=['Year', 'pos', 'player_name'], inplace=True)
    #Again this data is somewhat normalized now
    # save as csv
    normalized.to_csv('data_stat_minus_mean_normalized.csv', index=False)

    #have stats as their z score
    normalized = all_data.groupby(['pos', 'Year']).transform(lambda x: (x - x.mean()) / x.std())
    normalized = normalized.join(pos_name_year, how='outer')
    # label the all stars as 1 using left join
    normalized = pd.merge(normalized, all_stars, how='left', on=['Year', 'player_name'])
    # check that I didn't miss any
    print(normalized['is_all_star'].sum(skipna=True))
    print(all_stars.shape[0])
    # set the nan to 0 for all stats,
    normalized.fillna(value=0, inplace=True)
    normalized.drop(columns=['Year', 'pos', 'player_name'], inplace=True)
    # Again this data is somewhat normalized now
    # save as csv
    normalized.to_csv('data_z_score_normalized.csv', index=False)





if __name__ == "__main__":
    main()