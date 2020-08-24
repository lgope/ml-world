#include <stdio.h>
#define A 1
#define B 2
#define C 3
#define D 4
#define F 5
#define G 6
#define J 7
#define K 8
#define M 9
#define O 10
#define P 11

#define move_forward 12
#define move_right 13
#define move_left 14

#define REACHED 15

/*Agent program based on agent function. This C function
will take the percepts as a parameter and return the action*/

int next_move(int location)
{
    if (location == A)
    {
        return move_forward;
    }

    if (location == B)
    {
        return move_left;
    }

    if (location == C)
    {
        return move_left;
    }

    if (location == D)
    {
        return move_forward;
    }

    if (location == F)
    {
        return move_right;
    }

    if (location == G)
    {
        return move_right;
    }

    if (location == J)
    {
        return move_left;
    }

    if (location == K)
    {
        return move_forward;
    }

    if (location == M)
    {
        return move_right;
    }

    if (location == O)
    {
        return move_right;
    }

    if (location == P)
    {
        return REACHED;
    }
}

int main()
{
    //taking percepts
    int location;

    while (1)
    {

        printf("Please enter location - 1 if A/2 if B/3 if C/4 if D/5 if F/6 if G/7 if J/8 if K/9 if M/10 if O/11 if P :");
        scanf("%d", &location);

        //decision making
        //calling the C function that implemented agent function
        int action = next_move(location);

        //performing action

        if (action == 12)
        {
            printf("Move Forward\n");
        }

        if (action == 13)
        {
            printf("Move Right\n");
        }

        if (action == 14)
        {
            printf("Move Left\n");
        }

        if (action == 15)
        {
            printf("REACHED!\n");
            break;
        }
    }

    return 0;
}
