import pickle 
  
  
print("GFG") 
  
def write_file(): 
  
    f = open("travel.txt", "wb") 
    op = 'y'
  
    while op == 'y': 
  
        Travelcode = int(input("enter the travel id")) 
        Place = input("Enter the Place") 
        Travellers = int(input("Enter the number of travellers")) 
        buses = int(input("Enter the number of buses")) 
  
        pickle.dump([Travelcode, Place, Travellers, buses], f) 
        op = input("Dp you want to continue> (y or n)") 
  
    f.close() 
  
  
print("entering the details of passengers in the pickle file") 
write_file()