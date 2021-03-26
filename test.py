from contextlib import suppress


def external_code():
    for i in range(10):
        if i == 7:
            raise ValueError("I don't like sevens.")
        print(i)


with suppress(ValueError):
    external_code()
