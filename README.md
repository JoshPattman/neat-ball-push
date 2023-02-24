# Teaching an agent to push a ball to a target with the NEAT algorithm
## How the robot is trained
The robot uses a NEAT network as its brain. The network can include recurrent (backwards) connections, which allow it to have a primitive type of memory. These are depicted as green and yellow lines in the picture, whereas normal connections are blue and red.

Each generation, I simulate each robot on 5 differnt scenarios, where each scenario the robot and target start at a random position. The robot will always start the same distance from the target and the target will be the same distance from the centre. These scenarios last 10 seconds of simulated time, and after the ten seconds, a fitness for that scenario is calculated, which incentivises three different steps: touch the ball, push the ball to the target, run away from the target. You can see more about how this fitness function works in the code.

The inputs to the network are the normalised x and y distance to the ball from the robot, the normalised x and y distance to the centre from the robot, an indicator whether the ball has reached the target or not, and a bias value that is always 1. I use linear activations on input nodes, tanh on output nodes, and a variation of ReLU for hidden nodes, which has equation `ln(x+1)` when `x>0`. I found this prevents the recurrent connections creating very large value in the network.

## Run the code yourself
I have only ever tested this on linux, but it should run on other OSs. To run a new training run `go run . -n <name>`, which generates the networks to `nets/<name>`. To run a previously trained network run `go run . -n <name> -s`, which will load the named network and run a window. To reset the state whilst in the window press _s_.