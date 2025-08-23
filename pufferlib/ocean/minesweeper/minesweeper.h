#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "raylib.h"

#define HEIGHT 10
#define WIDTH 10
#define BOMBS 10

#define UP 0
#define DOWN 1
#define LEFT 2
#define RIGHT 3
#define MARK 4
#define REVEAL 5
#define NOOP 6
#define TOGGLE_COVER 7

#define HEADER_OFFSET 75
#define MIN_DIST 1
#define BOMBS_MULTIPLIER 2

const unsigned char EMPTY = '0';
const unsigned char COVERED = '9';
const unsigned char AGENT = 'a';
const unsigned char BOMB = 'b';
const unsigned char MARKED = 'm';

const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_LINES = (Color){50, 50, 50, 255};
const Color PUFF_BLUE = (Color){51, 51, 255, 255};
const Color PUFF_GREEN = (Color){0, 255, 0, 255};
const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_DARK_BLUE = (Color){0, 0, 153, 255};
const Color PUFF_BROWN = (Color){153, 76, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_PURPLE = (Color){153, 0, 153, 255};
const Color PUFF_GRAY = (Color){150, 150, 150, 255};

const Color* cell_colors[] = {
    &PUFF_BLUE,
    &PUFF_GREEN,
    &PUFF_RED,
    &PUFF_DARK_BLUE,
    &PUFF_BROWN,
    &PUFF_CYAN,
    &PUFF_PURPLE,
    &PUFF_GRAY
};

typedef struct {
    Texture2D puffer;
    Texture2D star;
} Client;

// node struct used to implement queue for reveal function
typedef struct node_t {
    int cell_index;
    struct node_t* next_node;  
} node_t;

// Required struct. Only use floats!
typedef struct {
    float perf;                 // Recommended 0-1 normalized single real number perf metric
    float score;                // Recommended unnormalized single real number perf metric
    float episode_return;       // Recommended metric: sum of agent rewards over episode
    float episode_length;       // Recommended metric: number of steps of agent episode
    
    // Any extra fields you add here may be exported to Python in binding.c
    float bomb_placed;
    
    float n; // Required as the last field 
} Log;

// Required that you have some struct for your env
// Recommended that you name it the same as the env file
typedef struct {
    Client* client;
    Log log;                        // Required field. Env binding code uses this to aggregate logs
    unsigned char* observations;    // Required. You can use any obs type, but make sure it matches in Python!
    int* actions;                   // Required. int* for discrete/multidiscrete, float* for box
    float* rewards;                 // Required
    unsigned char* terminals;       // Required. We don't yet have truncations as standard yet
    unsigned char* game_grid;
    int* bomb_positions;
    struct node_t* queue_head;
    struct node_t* current_head;
    struct node_t* queue_tail;
    int size;
    int tick;
    int r;
    int c;
    int num_bombs;
    int marked_bombs;
    char under_agent;
    int show_bombs;
    unsigned char* backup_obs;
    int reveal_number;
    int game_over;
    int total_cells;
    int discovered_cells;
} Minesweeper;

void add_log(Minesweeper* env) {
    env->log.perf = (float) env->discovered_cells / (env->total_cells - env->num_bombs);
    env->log.score = (float) env->discovered_cells;
    env->log.episode_length += env->tick;
    env->log.episode_return += env->rewards[0];
    env->log.bomb_placed = (float) env->marked_bombs / env->num_bombs;
    env->log.n++;
}

// given a bomb position update the count of surrounding cells
void update_bomb(Minesweeper* env, int pos) {
    if (pos % env->size == 0) { // first column
        if (pos == 0) { // upper left corner
            if (env->game_grid[pos + 1] != BOMB) env->game_grid[pos + 1]++;
            if (env->game_grid[pos + env->size + 1] != BOMB) env->game_grid[pos + env->size + 1]++;
            if (env->game_grid[pos + env->size] != BOMB) env->game_grid[pos + env->size]++;
        } else if (pos / env->size == env->size - 1) { // lower left corner
            if (env->game_grid[pos + 1] != BOMB) env->game_grid[pos + 1]++;
            if (env->game_grid[pos - env->size + 1] != BOMB) env->game_grid[pos - env->size + 1]++;
            if (env->game_grid[pos - env->size] != BOMB) env->game_grid[pos - env->size]++;
        } else { // rest
            if (env->game_grid[pos - env->size] != BOMB) env->game_grid[pos - env->size]++;
            if (env->game_grid[pos - env->size + 1] != BOMB) env->game_grid[pos - env->size + 1]++;
            if (env->game_grid[pos + 1] != BOMB) env->game_grid[pos + 1]++;
            if (env->game_grid[pos + env->size + 1] != BOMB) env->game_grid[pos + env->size + 1]++;
            if (env->game_grid[pos + env->size] != BOMB) env->game_grid[pos + env->size]++;
        }
    } else if (pos % env->size == env->size -1) { // last column
        if (pos == env->size - 1) { // upper right corner
            if (env->game_grid[pos - 1] != BOMB) env->game_grid[pos - 1]++;
            if (env->game_grid[pos + env->size - 1] != BOMB) env->game_grid[pos + env->size - 1]++;
            if (env->game_grid[pos + env->size] != BOMB) env->game_grid[pos + env->size]++;
        } else if (pos == env->size*env->size - 1) { // lower right corner
            if (env->game_grid[pos - env->size] != BOMB) env->game_grid[pos - env->size]++;
            if (env->game_grid[pos - env->size - 1] != BOMB) env->game_grid[pos - env->size - 1]++;
            if (env->game_grid[pos - 1] != BOMB) env->game_grid[pos - 1]++;
        } else { // rest
            if (env->game_grid[pos - env->size] != BOMB) env->game_grid[pos - env->size]++;
            if (env->game_grid[pos - env->size - 1] != BOMB) env->game_grid[pos - env->size - 1]++;
            if (env->game_grid[pos - 1] != BOMB) env->game_grid[pos - 1]++;
            if (env->game_grid[pos + env->size - 1] != BOMB) env->game_grid[pos + env->size - 1]++;
            if (env->game_grid[pos + env->size] != BOMB) env->game_grid[pos + env->size]++;
        }
    } else {
        if (pos / env->size == 0) { // first row
            if (env->game_grid[pos - 1] != BOMB) env->game_grid[pos - 1]++;
            if (env->game_grid[pos + env->size - 1] != BOMB) env->game_grid[pos + env->size - 1]++;
            if (env->game_grid[pos + env->size] != BOMB) env->game_grid[pos + env->size]++;
            if (env->game_grid[pos + env->size + 1] != BOMB) env->game_grid[pos + env->size + 1]++;
            if (env->game_grid[pos + 1] != BOMB) env->game_grid[pos + 1]++;
        } else if (pos /env->size == env->size - 1) { // last row
            if (env->game_grid[pos + 1] != BOMB) env->game_grid[pos + 1]++;
            if (env->game_grid[pos - env->size + 1] != BOMB) env->game_grid[pos - env->size + 1]++;
            if (env->game_grid[pos - env->size] != BOMB) env->game_grid[pos - env->size]++;
            if (env->game_grid[pos - env->size - 1] != BOMB) env->game_grid[pos - env->size - 1]++;
            if (env->game_grid[pos - 1] != BOMB) env->game_grid[pos - 1]++;
        } else { // all
            if (env->game_grid[pos - 1] != BOMB) env->game_grid[pos - 1]++;
            if (env->game_grid[pos + env->size - 1] != BOMB) env->game_grid[pos + env->size - 1]++;
            if (env->game_grid[pos + env->size] != BOMB) env->game_grid[pos + env->size]++;
            if (env->game_grid[pos + env->size + 1] != BOMB) env->game_grid[pos + env->size + 1]++;
            if (env->game_grid[pos + 1] != BOMB) env->game_grid[pos + 1]++;
            if (env->game_grid[pos - env->size + 1] != BOMB) env->game_grid[pos - env->size + 1]++;
            if (env->game_grid[pos - env->size] != BOMB) env->game_grid[pos - env->size]++;
            if (env->game_grid[pos - env->size - 1] != BOMB) env->game_grid[pos - env->size - 1]++;
        }
    }
}

// check number of marked bombs.
// return true if #marked >= cell number
int check_marked_bombs(Minesweeper* env, int pos) {
    if (env->game_grid[pos] == EMPTY)
        return true;
    int cell_num_bombs = env->game_grid[pos] - EMPTY;
    int marked_bombs = 0;

    if (pos % env->size == 0) { // first column
        if (pos == 0) { // upper left corner
            if (env->observations[pos + 1] == MARKED) marked_bombs++;
            if (env->observations[pos + env->size + 1] == MARKED) marked_bombs++;
            if (env->observations[pos + env->size] == MARKED) marked_bombs++;
        } else if (pos / env->size == env->size - 1) { // lower left corner
            if (env->observations[pos + 1] == MARKED) marked_bombs++;
            if (env->observations[pos - env->size + 1] == MARKED) marked_bombs++;
            if (env->observations[pos - env->size] == MARKED) marked_bombs++;
        } else { // rest
            if (env->observations[pos - env->size] == MARKED) marked_bombs++;
            if (env->observations[pos - env->size + 1] == MARKED) marked_bombs++;
            if (env->observations[pos + 1] == MARKED) marked_bombs++;
            if (env->observations[pos + env->size + 1] == MARKED) marked_bombs++;
            if (env->observations[pos + env->size] == MARKED) marked_bombs++;
        }
    } else if (pos % env->size == env->size -1) { // last column
        if (pos == env->size - 1) { // upper right corner
            if (env->observations[pos - 1] == MARKED) marked_bombs++;
            if (env->observations[pos + env->size - 1] == MARKED) marked_bombs++;
            if (env->observations[pos + env->size] == MARKED) marked_bombs++;
        } else if (pos == env->size*env->size - 1) { // lower right corner
            if (env->observations[pos - env->size] == MARKED) marked_bombs++;
            if (env->observations[pos - env->size - 1] == MARKED) marked_bombs++;
            if (env->observations[pos - 1] == MARKED) marked_bombs++;
        } else { // rest
            if (env->observations[pos - env->size] == MARKED) marked_bombs++;
            if (env->observations[pos - env->size - 1] == MARKED) marked_bombs++;
            if (env->observations[pos - 1] == MARKED) marked_bombs++;
            if (env->observations[pos + env->size - 1] == MARKED) marked_bombs++;
            if (env->observations[pos + env->size] == MARKED) marked_bombs++;
        }
    } else {
        if (pos / env->size == 0) { // first row
            if (env->observations[pos - 1] == MARKED) marked_bombs++;
            if (env->observations[pos + env->size - 1] == MARKED) marked_bombs++;
            if (env->observations[pos + env->size] == MARKED) marked_bombs++;
            if (env->observations[pos + env->size + 1] == MARKED) marked_bombs++;
            if (env->observations[pos + 1] == MARKED) marked_bombs++;
        } else if (pos /env->size == env->size - 1) { // last row
            if (env->observations[pos + 1] == MARKED) marked_bombs++;
            if (env->observations[pos - env->size + 1] == MARKED) marked_bombs++;
            if (env->observations[pos - env->size] == MARKED) marked_bombs++;
            if (env->observations[pos - env->size - 1] == MARKED) marked_bombs++;
            if (env->observations[pos - 1] == MARKED) marked_bombs++;
        } else { // all
            if (env->observations[pos - 1] == MARKED) marked_bombs++;
            if (env->observations[pos + env->size - 1] == MARKED) marked_bombs++;
            if (env->observations[pos + env->size] == MARKED) marked_bombs++;
            if (env->observations[pos + env->size + 1] == MARKED) marked_bombs++;
            if (env->observations[pos + 1] == MARKED) marked_bombs++;
            if (env->observations[pos - env->size + 1] == MARKED) marked_bombs++;
            if (env->observations[pos - env->size] == MARKED) marked_bombs++;
            if (env->observations[pos - env->size - 1] == MARKED) marked_bombs++;
        }
    }

    // printf("Number of marked cells: %d/%d\n", marked_bombs, cell_num_bombs);
    return marked_bombs >= cell_num_bombs;
}

// check that the bomb placement if far 'MIN_DIST' cells from agent (all directions)
int safe_distance_agent_bomb(Minesweeper* env, int agent_pos, int pos) {
    for (int i = 1; i <= MIN_DIST; i++) {
        if ((pos >= agent_pos - i*env->size - i && pos <= i*agent_pos - env->size + i) ||
            (pos >= agent_pos - i && pos <= agent_pos + i) ||
            (pos >= agent_pos + i*env->size - i && pos <= agent_pos + i*env->size + i))
        return false;
    }
    return true;
}


// return true if a bomb in 'pos' overlaps with bombs already place
// used only during generation
int bomb_overlap(Minesweeper* env, int pos, int curr_len) {
    for (int i = 0; i < curr_len; i++) {
        if (pos == env->bomb_positions[i])
            return true;
    }
    return false;
}

// print game_grid to terminal for debug
void print_grid(Minesweeper* env) {
    printf("----------------\n");
    for (int i = 0; i < env->size; i++) {
        for (int j = 0; j < env->size; j++)
            if (i == env->r && j == env->c)
                printf("x ");
            else printf("%c ", env->game_grid[i*env->size + j]);
        printf("\n");
    }
    printf("Under agent: %c\n", env->under_agent);
    printf("----------------\n");
}

// print observations to terminal for debug
void print_obs(Minesweeper* env) {
    printf("----------------\n");
    for (int i = 0; i < env->size; i++) {
        for (int j = 0; j < env->size; j++)
            printf("%c ", env->observations[i*env->size + j]);
        printf("\n");
    }
    printf("----------------\n");
}

// check if a cell has already been visited during flood to clear
int visited(Minesweeper* env, int node_pos) {
    struct node_t* tmp = env->queue_head;
    while (tmp != NULL) {
        if (tmp->cell_index == node_pos)
            return true;
        else tmp = tmp->next_node;
    }
    return false;
}

// add cell to queue
void add_cell_to_reveal(Minesweeper* env, int cell_pos) {
    if (cell_pos < 0 ||
        cell_pos >= env->size*env->size ||
        visited(env, cell_pos) ||
        env->observations[cell_pos] == MARKED)
        return;

    struct node_t* node = (struct node_t*) malloc(sizeof(struct node_t));
    if (node == NULL)
        exit(1);
    
    node->cell_index = cell_pos;
    node->next_node = NULL;

    if (env->queue_head == NULL) {  // if head is null queue is empty
        env->queue_head = node;
        env->current_head = node;
        env->queue_tail = node;
    } else {
        env->queue_tail->next_node = node;
        env->queue_tail = node;
    }
}

// free all elements in the queue
void empty_queue(Minesweeper* env) {
    env->current_head = NULL;
    env->queue_tail = NULL;
    while (env->queue_head != NULL) {
        struct node_t* tmp = env->queue_head;
        env->queue_head = tmp->next_node;
        free(tmp);
    }
}

// starts from a cell (head of the queue) and clears everything that can be cleared
void propagate_reveal(Minesweeper* env) {
    while (env->current_head != NULL) {
        struct node_t* current = env->current_head;
        int idx = current->cell_index;
        // printf("%p - %d - %c\n", current, idx, env->game_grid[idx]);

        if (env->game_grid[idx] == BOMB)
            env->game_over = 1;

        if (env->game_grid[idx] == EMPTY) {
            if (idx % env->size == 0) { // first column
                add_cell_to_reveal(env, idx + env->size);
                add_cell_to_reveal(env, idx + env->size + 1);
                add_cell_to_reveal(env, idx + 1);
                add_cell_to_reveal(env, idx - env->size + 1);
                add_cell_to_reveal(env, idx - env->size);
            } else if (idx % env->size == env->size - 1) {
                add_cell_to_reveal(env, idx - env->size);
                add_cell_to_reveal(env, idx - env->size - 1);
                add_cell_to_reveal(env, idx - 1);
                add_cell_to_reveal(env, idx + env->size - 1);
                add_cell_to_reveal(env, idx + env->size);
            } else {
                add_cell_to_reveal(env, idx - 1);
                add_cell_to_reveal(env, idx + env->size - 1);
                add_cell_to_reveal(env, idx + env->size);
                add_cell_to_reveal(env, idx + env->size + 1);
                add_cell_to_reveal(env, idx + 1);
                add_cell_to_reveal(env, idx - env->size + 1);
                add_cell_to_reveal(env, idx - env->size);
                add_cell_to_reveal(env, idx - env->size - 1);
            }    
        } else if (env->game_grid[idx] > EMPTY && env->game_grid[idx] <= '8') { // numbered cell
            // clean once all cell around if enough marks
            if (env->reveal_number && check_marked_bombs(env, idx)) {
                if (idx % env->size == 0) { // first column
                    add_cell_to_reveal(env, idx + env->size);
                    add_cell_to_reveal(env, idx + env->size + 1);
                    add_cell_to_reveal(env, idx + 1);
                    add_cell_to_reveal(env, idx - env->size + 1);
                    add_cell_to_reveal(env, idx - env->size);
                } else if (idx % env->size == env->size - 1) {
                    add_cell_to_reveal(env, idx - env->size);
                    add_cell_to_reveal(env, idx - env->size - 1);
                    add_cell_to_reveal(env, idx - 1);
                    add_cell_to_reveal(env, idx + env->size - 1);
                    add_cell_to_reveal(env, idx + env->size);
                } else {
                    add_cell_to_reveal(env, idx - 1);
                    add_cell_to_reveal(env, idx + env->size - 1);
                    add_cell_to_reveal(env, idx + env->size);
                    add_cell_to_reveal(env, idx + env->size + 1);
                    add_cell_to_reveal(env, idx + 1);
                    add_cell_to_reveal(env, idx - env->size + 1);
                    add_cell_to_reveal(env, idx - env->size);
                    add_cell_to_reveal(env, idx - env->size - 1);
                }
                env->reveal_number = 0;
            }
        }

        if (env->observations[idx] == COVERED) {
            env->observations[idx] = env->game_grid[idx];
            env->discovered_cells++;
        }

        env->current_head = current->next_node;
    }
    empty_queue(env);
}

// Required function
void c_reset(Minesweeper* env) {
    env->total_cells = env->size*env->size;
    memset(env->observations, 0, env->total_cells*sizeof(unsigned char));
    
    if (env->game_grid == NULL)
        env->game_grid = (unsigned char*) calloc(env->size*env->size, sizeof(unsigned char));
    if (env->game_grid == NULL)
        exit(-1);
    memset(env->game_grid, 0, env->total_cells*sizeof(unsigned char));

    if (env->backup_obs == NULL)
        env->backup_obs = (unsigned char*) calloc(env->size*env->size, sizeof(unsigned char));
    if (env->backup_obs == NULL)
        exit(-1);
    memset(env->backup_obs, 0, env->total_cells*sizeof(unsigned char));

    env->tick = 0;
    env->num_bombs = (int) sqrt(env->total_cells) * BOMBS_MULTIPLIER;
    if (env->bomb_positions == NULL)
        env->bomb_positions = (int*) calloc(env->num_bombs, sizeof(int));
    if (env->bomb_positions == NULL)
        exit(-1);
    memset(env->bomb_positions, 0, env->num_bombs*sizeof(int));
    env->marked_bombs = 0;
    int agent_start_position = rand() % env->total_cells;
    env->show_bombs = 0;
    env->reveal_number = 0;
    env->game_over = 0;
    env->discovered_cells = 1; // agent starting cell

    for (int i = 0; i < env->size; i++)
        for (int j = 0; j < env->size; j++)
            env->game_grid[i*env->size + j] = EMPTY;

    // place bombs
    int generated_bombs = 0;
    do {
            int bomb_pos = rand() % env->total_cells;
            if (safe_distance_agent_bomb(env, agent_start_position, bomb_pos)
                && !bomb_overlap(env, bomb_pos, generated_bombs)) {
                env->game_grid[bomb_pos] = BOMB;
                env->bomb_positions[generated_bombs++] = bomb_pos;
            }
    } while (generated_bombs < env->num_bombs);

    // update cells with nearby bombs count
    for (int i = 0; i < env->num_bombs; i++) {
        update_bomb(env, env->bomb_positions[i]);
    }

    // cover all map
    for (int i = 0; i < env->size; i++) {
        for (int j = 0; j < env->size; j++) {
            env->observations[i*env->size + j] = COVERED;
        }
    }

    // place puffer agent
    env->observations[agent_start_position] = AGENT;
    env->r = agent_start_position / env->size;
    env->c = agent_start_position % env->size;
    env->under_agent = EMPTY;

    // reveal cells from agents position
    add_cell_to_reveal(env, agent_start_position);
    propagate_reveal(env);
}

// Required function
void c_step(Minesweeper* env) {
    env->tick += 1;

    int action = env->actions[0];
    env->terminals[0] = 0;
    env->rewards[0] = 0;

    float reward = 0.0f;
    int init_free = env->discovered_cells;
    
    if (action == NOOP)
        return;

    env->observations[env->r*env->size + env->c] = env->under_agent;

    if (action == DOWN && env->r < env->size - 1) {
        env->r += 1;
    } else if (action == RIGHT && env->c < env->size - 1) {
        env->c += 1;
    } else if (action == UP && env->r > 0) {
        env->r -= 1;
    } else if (action == LEFT && env->c > 0) {
        env->c -= 1;
    } else if (action == MARK) {
        if (env->observations[env->r*env->size + env->c] == COVERED) {
            // printf("Marking cell: (%d-%d)\n", env->r, env->c);
            env->observations[env->r*env->size + env->c] = MARKED;
            env->marked_bombs++;
        } else if (env->observations[env->r*env->size + env->c] == MARKED) {
            // printf("Removing mark cell: (%d-%d)\n", env->r, env->c);
            env->observations[env->r*env->size + env->c] = COVERED;
            env->marked_bombs--;
        }
    } else if (action == REVEAL) {
        if (env->observations[env->r*env->size + env->c] == COVERED) {
            env->discovered_cells++;
            // printf("Revealing cell: (%d-%d)\n", env->r, env->c);
            if (env->game_grid[env->r*env->size + env->c] == BOMB)
                env->game_over = 1;
            env->observations[env->r*env->size + env->c] = env->game_grid[env->r*env->size + env->c];
        } else if (env->observations[env->r*env->size + env->c] > EMPTY &&
                env->observations[env->r*env->size + env->c] < '8') {
            // printf("Reveal on number cell\n");
            env->reveal_number = 1;
            add_cell_to_reveal(env, env->r*env->size + env->c);
            propagate_reveal(env);
        }
    } else if (action == TOGGLE_COVER) {
        if (env->show_bombs == 0) {
            for (int i = 0; i < env->size*env->size; i++) {
                env->backup_obs[i] = env->observations[i];
                env->observations[i] = env->game_grid[i];
            }
            env->show_bombs = 1;
        } else {
            for (int i = 0; i < env->size*env->size; i++)
                env->observations[i] = env->backup_obs[i];
            env->show_bombs = 0;
        }
    }

    reward = (float) (env->discovered_cells - init_free) / (env->total_cells - env->num_bombs - init_free);
    env->rewards[0] = reward;

    if (env->discovered_cells == env->total_cells - env->num_bombs) { // win
        env->terminals[0] = 1;
        env->rewards[0] = 1.0;
        add_log(env);
        c_reset(env);
        return;
    }

    if (env->game_over) {
        env->terminals[0] = 1;
        env->rewards[0] = -1.0;
        add_log(env);
        c_reset(env);
        return;
    }

    int pos = env->r*env->size + env->c;
    env->under_agent = env->observations[pos];
    env->observations[pos] = AGENT;
}

// Required function. Should handle creating the client on first call
void c_render(Minesweeper* env) {
    if (env->client == NULL) {
        InitWindow(64*env->size, 64*env->size + HEADER_OFFSET, "PufferLib Minesweeper");
        SetTargetFPS(5);

        env->client = (Client*) calloc(1, sizeof(Client));
        // Don't do this before calling InitWindow
        env->client->puffer = LoadTexture("resources/shared/puffers_128.png");
        env->client->star = LoadTexture("resources/minesweeper/star.png");
    }

    // Standard across our envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    int px = 64;

    // printing score board
    DrawText(
        TextFormat(
            "Step: %d\nBomb Remaining: %d/%d\nCells: %d/%d",
            env->tick,
            env->num_bombs - env->marked_bombs,
            env->num_bombs,
            env->discovered_cells,
            env->total_cells - env->num_bombs
        ),
        10, 10, 20, PUFF_WHITE
    );

    // printing minegrid
    for (int i = 0; i < env->size; i++) {
        for (int j = 0; j < env->size; j++) {
            int tex = env->observations[i*env->size + j];

            int screen_x = j * px;
            int screen_y = i * px + HEADER_OFFSET;

            Rectangle cell_rect = {
                .x = screen_x,
                .y = screen_y,
                .width = px,
                .height = px
            };
            // Draw grid cell border
            DrawRectangleLines((int) cell_rect.x, (int) cell_rect.y, (int) cell_rect.width, (int) cell_rect.height, PUFF_LINES);

            if (tex == EMPTY)
                continue;
            
            if (tex == COVERED) {
                DrawRectangle(screen_x + px/8, screen_y + px/8, 3*px/4, 3*px/4, PUFF_GRAY);
            } else if (tex == MARKED) {
                DrawTexture(
                    env->client->star,
                    screen_x - 32,
                    screen_y - 32,
                    WHITE
                );
            } else if (tex == AGENT) {
                DrawTexturePro(
                    env->client->puffer,
                    (Rectangle) { 0, 0, 128, 128, },
                    (Rectangle){
                        screen_x, 
                        screen_y,
                        px,
                        px
                    },
                    (Vector2){0, 0},
                    0,
                    WHITE
                );
            } else if (tex >= '1' && tex <= '8') { // number tiles
                DrawText(
                    TextFormat("%d", tex - EMPTY),
                    screen_x + px/3,
                    screen_y + px/5,
                    50,
                    *cell_colors[tex - '1']
                );
            } else if (tex == BOMB) {
                DrawRectangle(screen_x + px/8, screen_y + px/8, 3*px/4, 3*px/4, PUFF_RED);
            } else {
                fprintf(stderr, "Error: Unknown value for tile(int): %d - idx (%d, %d)", tex, i, j);
                exit(-1);
            }
        }
    }
    EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(Minesweeper* env) {
    if (IsWindowReady()) {
        free(env->client);
        free(env->game_grid);
        free(env->bomb_positions);
        free(env->backup_obs);
        if (env->queue_head != NULL) empty_queue(env);
        CloseWindow();
    }
}