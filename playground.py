def coinChange(coins, amount):
    def coinChangeAl(coins, rem, count):
        if rem < 0:
                return -1
        if rem == 0:
            return 0
        if count[rem-1] != 0:
            return count[rem-1]

        min = float('inf')
        for coin in coins: 
            res = coinChangeAl(coins, rem - coin, count)
            if res >= 0 & res < min: 
                min = 1 + res
        count[rem - 1] = -1 if (min == float('inf')) else min
        return count[rem - 1 ]
    
    count = [0 for i in range(amount)]
    var = coinChangeAl(coins, amount, count)
    print(count)
    return var

coins = [1,2,5]
amount = 11

print(coinChange(coins, amount), '\n')

coins = [1,9,10]
amount = 27

print(coinChange(coins, amount))