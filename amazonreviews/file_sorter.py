import sys, getopt

def main(argv):
   inputfolder = ''
   sample = 0
   nlabel = "__label__1"
   plabel = "__label__2"
   
   
   try:
      opts, args = getopt.getopt(argv,"hi:s:",["ifile=","sample="])
   except getopt.GetoptError:
      print('test.py -i <inputfolder> -o <outputfile>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('test.py -i <inputfolder> -o <outputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfolder = arg
      elif opt in ("-s", "--sample"):
         sample = int(arg)
   
   
   if(len(inputfolder) > 0):
       splitFile(inputfolder, sample, nlabel, plabel)
   
def splitFile(inputfolder, samplesize, nlabel, plabel):
    inputfile = "./%s/%s.ft.txt" % (inputfolder, inputfolder)
    print('Input file is "%s"' % inputfolder)
    
    positivefile = "./%s/ordered_%s_pos.txt" % (inputfolder, inputfolder) 
    negativefile = "./%s/ordered_%s_neg.txt" % (inputfolder, inputfolder) 

    content = []
    positives = []
    negatives = []
    p=0
    n=0
    
    with open(inputfile, encoding="utf8") as f:
        content = f.readlines()

    if samplesize == 0:
        samplesize = len(content)
     
    for line in content:
        if(n < samplesize and line.startswith(nlabel)):
            n += 1
            negatives.append(line)
        elif(p < samplesize and line.startswith(plabel)):
            p += 1
            positives.append(line)
         
    with open(positivefile, "w+", encoding="utf8") as f:
        for line in positives:
            f.write('%s' % line)
            
    with open(negativefile, "w+", encoding="utf8") as f:
        for line in negatives:
            f.write('%s' % line)

   
if __name__ == "__main__":
   main(sys.argv[1:])
   
   
# https://www.tutorialspoint.com/python/python_command_line_arguments.htm