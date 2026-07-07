import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("company_sales_data.csv")


print(df.head(10))


plt.figure(figsize=(10, 5))
# plt.plot(df["month_number"],df['total_profit'], marker='o')
# plt.xlabel("Month Number")
# plt.ylabel("Total Profit")
# plt.title("Total Profit of All months")
# plt.grid(True)
# plt.show()

plt.subplot(2,1,1)
plt.plot(df['month_number'],df['bathingsoap'],color='green',marker='o')
plt.xticks(df['month_number'])
plt.ylabel("Units sold")
plt.title("Bathing Soap Sales Data")
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(df['month_number'],df['facewash'],color='red',marker='o')
plt.xlabel('month_number')
plt.ylabel("Units sold")
plt.title("Facewash Sales Data")
plt.grid(True)

plt.tight_layout()
plt.show()
