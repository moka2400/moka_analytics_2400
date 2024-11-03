# src/main.py

# from data_prep import load_and_print_data
from Methods.Examples.linear_reg import linear_regression_stocks
from Methods.Examples.monte_carlo_simulations import monte_carlo_simulations
from Methods.Football.pl_winner import predict_pl_winner

def main():
    print("Starting the Moka Analytics application...")
    print("Running")
    
    # linear_regression_stocks()
    # monte_carlo_simulations()
    predict_pl_winner()
    
    print("processing complete.")

if __name__ == "__main__":
    main()
