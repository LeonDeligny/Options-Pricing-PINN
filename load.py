import pandas as pd

def load_data():
    # Load the CSV file
    file_path = "/Users/leondeligny/Desktop/Option Pricing/tsla_2019_2022.csv"

    # Select relevant columns for input features (e.g., [UNDERLYING_LAST] and [DTE]) and target outputs (C_LAST and P_LAST)
    df = pd.read_csv(
        file_path, 
        usecols=[
            ' [C_IV]', ' [P_IV]', 
            ' [C_LAST]', ' [P_LAST]', 
            ' [UNDERLYING_LAST]', ' [DTE]'
        ],
        dtype={
            ' [C_IV]': 'float64', ' [P_IV]': 'float64', 
            ' [C_LAST]': 'float64', ' [P_LAST]': 'float64', 
            ' [UNDERLYING_LAST]': 'float64', ' [DTE]': 'float64'
        },
        low_memory=False,
        na_values=[' ', ''],
    )

    # Convert all columns to numeric if possible, forcing non-numeric values to NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    '''
    Here is a detailed explanation of what the dataframe df contains : 

    Index column, column name: definition of the content.
    1.	[QUOTE_UNIXTIME]: Unix timestamp of the option quote time, representing the number of seconds since January 1, 1970 (UTC).
    2.	[QUOTE_READTIME]: Human-readable format of the quote retrieval time.
    3.	[QUOTE_DATE]: Date of the option quote in a standard date format.
    4.	[QUOTE_TIME_HOURS]: Time of the option quote, typically in hours, to give an intra-day context.
    5.	[UNDERLYING_LAST]: Last traded price of the underlying asset at the time of the option quote.
    6.	[EXPIRE_DATE]: Expiration date of the option, indicating the last date the option can be exercised.
    7.	[EXPIRE_UNIX]: Unix timestamp of the option expiration date.
    8.	[DTE]: Days to expiration, calculated as the difference between the expiration date and the quote date.
    9.	[C_DELTA]: Delta of the call option, representing sensitivity of the option’s price to the underlying asset’s price.
    10.	[C_GAMMA]: Gamma of the call option, indicating the rate of change of delta with respect to the underlying asset’s price.
    11.	[C_VEGA]: Vega of the call option, measuring sensitivity of the option’s price to volatility changes in the underlying asset.
    12.	[C_THETA]: Theta of the call option, representing the sensitivity of the option’s price to the passage of time (time decay).
    13.	[C_RHO]: Rho of the call option, reflecting the sensitivity of the option’s price to changes in interest rates.
    14.	[C_IV]: Implied volatility of the call option, indicating the market’s expectation of volatility.
    15.	[C_VOLUME]: Trading volume of the call option on the quote date.
    16.	[C_LAST]: Last traded price of the call option.
    17.	[C_SIZE]: Size of the last call option trade (number of contracts).
    18.	[C_BID]: Bid price for the call option, indicating the highest price a buyer is willing to pay.
    19.	[C_ASK]: Ask price for the call option, indicating the lowest price a seller is willing to accept.
    20.	[STRIKE]: Strike price of the option, which is the predetermined price at which the option can be exercised.
    21.	[P_BID]: Bid price for the put option.
    22.	[P_ASK]: Ask price for the put option.
    23.	[P_SIZE]: Size of the last put option trade (number of contracts).
    24.	[P_LAST]: Last traded price of the put option.
    25.	[P_DELTA]: Delta of the put option.
    26.	[P_GAMMA]: Gamma of the put option.
    27.	[P_VEGA]: Vega of the put option.
    28.	[P_THETA]: Theta of the put option.
    29.	[P_RHO]: Rho of the put option.
    30.	[P_IV]: Implied volatility of the put option.
    31.	[P_VOLUME]: Trading volume of the put option on the quote date.
    32.	[STRIKE_DISTANCE]: Absolute difference between the underlying asset price and the option strike price.
    33.	[STRIKE_DISTANCE_PCT]: Strike distance as a percentage of the underlying asset’s price, indicating how far in or out of the money the option is.
    '''

    # Drop rows with missing values
    df = df.dropna()

    # Select a random quarter (25%) of the rows from the DataFrame
    df_sampled = df.sample(frac=0.25, random_state=42)

    return df_sampled

load_data()