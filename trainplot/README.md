This directory contains a slightly modified version of https://github.com/joeyhng/trainplot

# TrainPlotter

Generate JSON files for plots in during training networks.

## Usage
```lua
require 'TrainPlotter'
local plotter = TrainPlotter.new('plot-data/exp.json')
plotter:add('Accracy', 'Train', 1, 0.5)
plotter:add('Accracy', 'Train', 2, 0.7)
plotter:add('Accracy', 'Train', 3, 0.8)
plotter:add('Accracy', 'Test', 1, 0.45)
plotter:add('Accracy', 'Test', 2, 0.6)
plotter:add('Accracy', 'Test', 3, 0.7)
```

## Seeing Plots
Start a HTTP server by
```
cd /path/to/trainplot/
python -m SimpleHTTPServer
```

Open showplots.html in browser to change the JSON path, or specify with query string like
```
http://localhost:8000/showplots.html?path=plot-data/UCF101/exp.json
```


