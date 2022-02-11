
def id_generator():
    num = 0
    while True:
        yield num
        num += 1
