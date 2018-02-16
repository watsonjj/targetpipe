from itertools import chain, combinations


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def main():
    leds = list(range(10))
    led_combinations = list(powerset(leds))
    print(led_combinations)
    print(len(led_combinations))

if __name__ == '__main__':
    main()
