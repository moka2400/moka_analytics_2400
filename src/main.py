# src/main.py

from intro_stuff.pl_winner import predict_pl_winner
from co2_emissions.EDA.eda import run_eda

def main():
    print("Starting the Moka Analytics application...")
    print("Running")
    
    # predict_pl_winner()
    run_eda()
    
    print("processing complete.")

if __name__ == "__main__":
    main()
