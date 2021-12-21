#!/u3.bath/s32/adbb135
#SBATCH -D /users/adbb135/code_to_test		 # Working directory
#SBATCH --job-name=my_fluent                 # Job name
#SBATCH --mail-type=END,FAIL                 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=Xavier.Murrow@city.ac.uk # Where to send mail	
#SBATCH --exclusive                          # Exclusive use of nodes
#SBATCH --nodes=2                            # Run on 2 nodes (each node has 48 cores)
#SBATCH --ntasks-per-node=8                  # Use all the cores on each node
#SBATCH --mem=0                              # Expected memory usage (0 means use all available memory)
#SBATCH --time=00:05:00                      # Time limit hrs:min:sec
#SBATCH --gres=gpu:1						 #Use one gpu
#SBATCH --output=myfluent_test_%j.out        # Standard output and error log [%j is replaced with the jobid]
#SBATCH --error=myfluent_test_%j.error

#enable modules
source /opt/flight/etc/setup.sh
flight env activate gridware

#remove any unwanted modules
module purge

#Modules required
#This is an example you need to select the modules your code needs.
module load python/3.7.12
module load libs/nvidia-cuda/11.2.0/bin

#Run your script.
python3 test.py