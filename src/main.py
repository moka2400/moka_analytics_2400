# src/main.py

# from data_prep import load_and_print_data
from Methods.linear_reg import linear_regression_stocks

def main():
    print("Starting the Moka Analytics application...")
    print("Running linear regression")
    #load_and_print_data()

    linear_regression_stocks()
    print("processing complete.")

if __name__ == "__main__":
    main()
