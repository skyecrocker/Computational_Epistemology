def time_to_lock(choices, steps):
    last_choice = choices[0]
    flipped = 1 if last_choice == 0 else 0 

    if flipped in choices:
        time_to_lock = steps - choices.index(flipped)
    else:
        time_to_lock = 0

    return time_to_lock