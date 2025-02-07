import os
from collections import deque

# Define the characters representing empty cells and paths
EMPTY_CELL: str = "."

# Description of possible connections for each pipe symbol
PIPE_CONNECTIONS: dict[str, list[tuple[int, int]]] = {
    "|": [(-1, 0), (1, 0)],     # Up, down
    "-": [(0, -1), (0, 1)],     # Left, right
    "L": [(-1, 0), (0, 1)],     # Up, right
    "J": [(-1, 0), (0, -1)],    # Up, left
    "7": [(1, 0),  (0, -1)],    # Down, left
    "F": [(1, 0),  (0, 1)],     # Down, right
    EMPTY_CELL: [],             # No connections
}


def get_input_data(
        filename: str,
) -> list[list[str]] | None:
    """
    Reads input data from a file
    """

    if not os.path.exists(filename):
        print(f"Error: The file '{filename}' does not exist")
        return

    with open(filename, "r") as file:
        return [list(line) for line in file.read().strip().split("\n")]


def save_maze_to_file(
        maze: list[str],
        filename: str,
) -> None:
    """
    Saves the maze to a file
    """

    with open(filename, "w") as file:
        for row in maze:
            file.write(row + "\n")
    print(f"Maze saved to {filename}")


def find_start(
        maze: list[list[str]],
) -> tuple[int, int] | None:
    """
    Find the coordinates of 'S' in the maze
    """

    for row_index, row in enumerate(maze):
        for column_index, cell in enumerate(row):
            if cell == "S":
                return row_index, column_index


def get_neighbors(
        maze: list[list[str]],
        start_position: tuple[int, int],
) -> list[tuple[int, int]]:
    """
    Determine the actual neighbors for the start position 'S'
    """

    row, column = start_position

    neighbors: list[tuple[int, int]] = []
    num_rows = len(maze)
    num_columns = len(maze[0])

    for delta_row, delta_column in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # All 4 directions
        new_row = row + delta_row
        new_column = column + delta_column

        if (
            0 <= new_row < num_rows
            and 0 <= new_column < num_columns
            and maze[new_row][new_column] in PIPE_CONNECTIONS
            and (-delta_row, -delta_column) in PIPE_CONNECTIONS[maze[new_row][new_column]]
        ):
            neighbors.append((new_row, new_column))

    return neighbors


def determine_pipe(
        neighbors: list[tuple[int, int]],
        start_position: tuple[int, int],
) -> str | None:
    """
    Determine which pipe symbol should be at the 'S' position
    """

    start_row, start_column = start_position

    delta_row_1 = neighbors[0][0] - start_row
    delta_column_1 = neighbors[0][1] - start_column

    delta_row_2 = neighbors[1][0] - start_row
    delta_column_2 = neighbors[1][1] - start_column

    # Search for the first pipe_symbol that matches the conditions
    return next(
        (
            pipe_symbol
            for pipe_symbol, directions
            in PIPE_CONNECTIONS.items()
            if (
                (delta_row_1, delta_column_1) in directions
                and (delta_row_2, delta_column_2) in directions
            )
        ),
        None # pipe_symbol is not found => return None
    )


def get_loop_cells(
        maze: list[list[str]],
        start_position: tuple[int, int],
) -> set[tuple[int, int]]:
    """
    Determines all reachable cells from the start position in the maze using connected pipes.
    """
    queue = deque([start_position])
    visited_cells = {start_position}

    while queue:
        current_row, current_column = queue.popleft()  # Current cell coordinates

        for delta_row, delta_column in PIPE_CONNECTIONS[maze[current_row][current_column]]:
            # Calculate new cell coordinates
            new_row = current_row + delta_row
            new_column = current_column + delta_column

            if (new_row, new_column) not in visited_cells:
                visited_cells.add((new_row, new_column))
                queue.append((new_row, new_column))

    # Return the set of visited cells
    return visited_cells


def get_result(
        maze: list[list[str]],
        loop_cells: set[tuple[int, int]],
) -> int:
    """
    Counts the number of '.' cells that are inside the enclosed loop.
    """

    maze = [
        "".join(
            symbol if (row_index, column_index) in loop_cells else EMPTY_CELL
            for column_index, symbol in enumerate(row)
        )
        for row_index, row in enumerate(maze)
    ]

    save_maze_to_file(
        maze=maze,
        filename="result.txt",
    )

    outside: set = set()

    for row_index, row in enumerate(maze):
        within = False
        up = None
        for column_index, symbol in enumerate(row):
            match symbol:
                case "|":
                    if up is not None:
                        raise ValueError(
                            f"Unexpected state: 'up' should be None before symbol '{symbol}' "
                            f"| ({row_index}, {column_index})")
                    within = not within

                case "-":
                    if up is None:
                        raise ValueError(
                            f"Unexpected state: 'up' should not be None before symbol '{symbol}' "
                            f"| ({row_index}, {column_index})"
                        )

                case "L" | "F":
                    if up is not None:
                        raise ValueError(
                            f"Unexpected state: 'up' should be None before symbol '{symbol}' "
                            f"| ({row_index}, {column_index})"
                        )
                    up = symbol == "L"

                case "J" | "7":
                    if up is None:
                        raise ValueError(
                            f"Unexpected state: 'up' should not be None before symbol '{symbol}' "
                            f"| ({row_index}, {column_index})"
                        )
                    if symbol != ("J" if up else "7"):
                        within = not within
                    up = None

                case ".":
                    pass

                case _:
                    raise ValueError(f"Unexpected symbol: {symbol} | ({row_index}, {column_index})")

            if not within:
                outside.add((row_index, column_index))

    return len(maze) * len(maze[0]) - len(outside | loop_cells)  # Number of tiles


def solve_pipe_maze(
        maze: list[list[str]],
) -> tuple[int | None, int | None]:
    start_position = find_start(
        maze=maze,
    )
    if start_position is None:
        print("Error: 'S' not found in the input data")
        return None, None

    neighbors = get_neighbors(
        maze=maze,
        start_position=start_position,
    )
    if len(neighbors) != 2:
        print(f"Error: 'S' has {len(neighbors)} neighbors, but it should have exactly 2")
        return None, None

    correct_pipe_symbol = determine_pipe(
        neighbors=neighbors,
        start_position=start_position,
    )
    if not correct_pipe_symbol:
        print("Error: Unable to determine the correct pipe shape for 'S'")
        return None, None

    print(f"Start position: {start_position}")
    print(f"Neighbors of S: {neighbors}")
    print(f"Pipe for S: '{correct_pipe_symbol}'")

    # Update 'S' to the correct pipe symbol
    start_row, start_column = start_position
    maze[start_row][start_column] = correct_pipe_symbol

    loop_cells = get_loop_cells(
        maze=maze,
        start_position=start_position,
    )

    max_distance = len(loop_cells) // 2
    tiles_number = get_result(
        maze=maze,
        loop_cells=loop_cells,
    )

    return max_distance, tiles_number


if __name__ == "__main__":
    # Get input data
    # file_name = input("Enter the input file name: ").strip()
    file_name = "input_puzzle.txt"
    maze_input = get_input_data(filename=file_name)

    p1_result, p2_result = solve_pipe_maze(maze_input)

    if p1_result is not None:
        print(
            f"The maximum distance from the starting point to the farthest reachable point along the loop: {p1_result}"
        )

    if p2_result:
        print(f"The total number of tiles that are enclosed within the loop: {p2_result}")
