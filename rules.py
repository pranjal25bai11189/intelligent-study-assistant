def give_advice(focus, sleep, time):
    if focus == 0:
        if sleep < 6:
            return "You are sleep deprived. Take rest before studying."
        elif time == "afternoon":
            return "You may feel sleepy. Take a short break and revise easy topics."
        else:
            return "Avoid heavy topics. Focus on revision."
    else:
        if sleep >= 7:
            return "Great condition! Study difficult topics now."
        else:
            return "You are focused, but improve sleep for better performance."