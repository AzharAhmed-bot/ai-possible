# Topic 2 — Data Visualization

A tour of the main chart types in **matplotlib**, using small everyday datasets
(election results, sales, profit, exam scores). The goal is to learn *which
chart to use for which kind of data*.

## What it does

Builds 8 different charts from scratch, each demonstrating a common plot type
and when to reach for it.

## Plots (in [`plots/`](plots/))

| File            | Chart type          | What it shows / when to use it                                  |
|-----------------|---------------------|-----------------------------------------------------------------|
| `figure_01.png` | Pie chart           | 2017 UK election vote share — parts of a whole.                 |
| `figure_02.png` | Line + scatter      | First 10 Fibonacci numbers — a trend plus its exact points.     |
| `figure_03.png` | Line with fill      | Monthly profit — a value changing over time.                    |
| `figure_04.png` | Pie chart           | Units sold per product — proportions across categories.         |
| `figure_05.png` | Styled line         | Same profit data with custom colours, markers and line style.   |
| `figure_06.png` | Grouped bar chart   | Face Cream vs Face Wash sales — comparing two series per month.  |
| `figure_07.png` | Histogram           | Profit distribution — which value ranges happen most often.     |
| `figure_08.png` | Box plot            | Student scores — median, spread and outliers at a glance.       |

## Quick guide: which chart when?

- **Pie** → parts of a whole (percentages that add up to 100%).
- **Line** → something changing over time or order.
- **Bar** → comparing categories side by side.
- **Histogram** → the shape/spread of a single set of numbers.
- **Box plot** → median, quartiles and outliers in one picture.

## Run it

```bash
python3 Topic2_Visualization.py
```

Plots are saved to `plots/`.
