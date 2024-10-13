# src/main.py

# from data_prep import load_and_print_data
from Methods.linear_reg import linear_regression_stocks
from Methods.monte_carlo_simulations import monte_carlo_simulations

def main():
    print("Starting the Moka Analytics application...")
    print("Running monte carlo simulations")
    
    #load_and_print_data()
    #linear_regression_stocks()
    monte_carlo_simulations()
    
    print("processing complete.")

if __name__ == "__main__":
    main()
