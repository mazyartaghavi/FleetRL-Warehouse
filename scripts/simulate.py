import sys
from fleetrl_warehouse.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "simulate")
    main()
