import sys

#if __name__ == "__main__":
#    print(f"Arguments count: {len(sys.argv)}")
#    for i, arg in enumerate(sys.argv):
#        print(f"Argument {i:>6}: {arg}")

if (len(sys.argv) <= 1):
    print("Missing Process Type Arguement")
elif(sys.argv[1] != "-train" and sys.argv[1] != "-infer"):
        print(sys.argv[1] + ": Command not part of the intent_analysis module.")
else:
    if(sys.argv[1]=="-train"):
        import intent_analysis_train
    else:
        import intent_analysis_infer
