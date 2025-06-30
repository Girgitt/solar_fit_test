Simple polynomial fitting of sensor data to match solar irradiance values derived from a luxometer with reference measurements from a calibrated irradiance meter.

Commands to execute the program:  
python main.py --action=update --model_id=your_id --csv=./data/your_file.csv  
python main.py --action=execute --model_id=your_id --csv=./data/your_file.csv    

project/  
├── main.py               # Main execution script for training or inference  
├── utils.py              # All utilities functions  
├── plots/                # Generated plots (e.g. predictions, raw series)  
├── logs/                 # JSON files storing model coefficients or checkpoints  
├── data/                 # CSV files with data  
└── README.md             # This documentation file  
