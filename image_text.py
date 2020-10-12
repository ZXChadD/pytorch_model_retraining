




def main():
    file = open("trainval.txt", "w+")
    for i in range(200):
        file.write(str(i) + '\n')


if __name__ == '__main__':
    main()