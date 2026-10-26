/* Offline AUV navigation environment. No DOM, network, or global random state.
 * The hand-written map-aware baseline is separate from learned TD3 actions.
 * Dynamics, observations, rewards, and terminal events match environment.py;
 * seeded courses use a JavaScript RNG, not NumPy's bit sequence.
 */
(function (root) {
  "use strict";

  const TAU = 2 * Math.PI;
  const DIAGONAL = Math.hypot(20, 20);
  const CONSTANTS = Object.freeze({
    WORLD_WIDTH: 20, WORLD_HEIGHT: 20, VEHICLE_RADIUS: 0.25,
    GOAL_RADIUS: 0.75, MAX_SPEED: 1.8, MAX_YAW_RATE: 1.2,
    ACCELERATION: 0.9, DRAG: 0.3, SONAR_RANGE: 6,
    SONAR_RAYS: 16, OBSERVATION_DIM: 21, MAX_TRAIL: 600,
  });
  const MAPS = {
    open: [],
    reef: [[7, 7.7, 1], [11.3, 10.6, 1.15], [13.4, 15.1, 0.85],
      [5.1, 12.3, 0.8], [15.7, 7.2, 0.9]],
    channel: [[5.8, 9.4, 1.2], [8.8, 5.5, 1.2], [9.1, 13.1, 1.2],
      [12.1, 8.9, 1.2], [12.5, 16.7, 1.2], [15.5, 12.4, 1.2]],
  };
  const clip = (value, lower, upper) => Math.max(lower, Math.min(upper, value));
  function wrapAngle(angle) {
    // JS remainder is signed; this reproduces Python's positive modulus.
    return ((angle + Math.PI) % TAU + TAU) % TAU - Math.PI;
  }
  function validateSeed(seed) {
    if (!Number.isSafeInteger(seed) || seed < 0) {
      throw new RangeError("seed must be a nonnegative safe integer");
    }
  }
  function emptyReward() {
    return {progress: 0, goal_bonus: 0, collision_penalty: 0,
      time_cost: 0, control_cost: 0, proximity_cost: 0};
  }

  /** Local Mulberry32 stream with checkpointable Box-Muller normal state. */
  class SeededRNG {
    constructor(seed = 7) {
      validateSeed(seed);
      this._state = seed >>> 0;
      this._normalSpare = null;
    }
    _next() {
      this._state = (this._state + 0x6D2B79F5) >>> 0;
      let value = this._state;
      value = Math.imul(value ^ (value >>> 15), value | 1);
      value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
      return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    }
    uniform(low = 0, high = 1) {
      if (!Number.isFinite(low) || !Number.isFinite(high) || high < low
          || !Number.isFinite(high - low)) {
        throw new RangeError("uniform bounds must be finite and ordered");
      }
      return low + (high - low) * this._next();
    }
    normal() {
      if (this._normalSpare !== null) {
        const result = this._normalSpare;
        this._normalSpare = null;
        return result;
      }
      const radius = Math.sqrt(-2 * Math.log(1 - this._next()));
      const angle = TAU * this._next();
      this._normalSpare = radius * Math.sin(angle);
      return radius * Math.cos(angle);
    }
    integer(max) {
      if (!Number.isSafeInteger(max) || max <= 0 || max > 4294967296) {
        throw new RangeError("integer max must be in [1, 2^32]");
      }
      // Rejection avoids bias when max does not divide 2^32.
      const limit = Math.floor(4294967296 / max) * max;
      let value;
      do { value = Math.floor(this._next() * 4294967296); } while (value >= limit);
      return value % max;
    }
    getState() {
      return {algorithm: "mulberry32-v1", state: this._state,
        normal_spare: this._normalSpare};
    }
    setState(state) {
      if (!state || state.algorithm !== "mulberry32-v1"
          || !Number.isInteger(state.state) || state.state < 0 || state.state > 0xFFFFFFFF
          || !(state.normal_spare === null || Number.isFinite(state.normal_spare))) {
        throw new RangeError("invalid random generator state");
      }
      this._state = state.state >>> 0;
      this._normalSpare = state.normal_spare;
      return this;
    }
  }

  function segmentCircleFraction(sx, sy, ex, ey, cx, cy, radius) {
    const ox = sx - cx, oy = sy - cy, dx = ex - sx, dy = ey - sy;
    const c = ox * ox + oy * oy - radius * radius;
    if (c <= 1e-12) return 0;
    const a = dx * dx + dy * dy;
    if (a <= 1e-24) return null;
    const b = 2 * (ox * dx + oy * dy);
    const discriminant = b * b - 4 * a * c;
    const tolerance = 1e-12 * Math.max(1, b * b, Math.abs(4 * a * c));
    if (discriminant < -tolerance) return null;
    const root = Math.sqrt(Math.max(0, discriminant));
    const first = (-b - root) / (2 * a);
    if (first >= -1e-12 && first <= 1 + 1e-12) return clip(first, 0, 1);
    const second = (-b + root) / (2 * a);
    return second >= -1e-12 && second <= 1 + 1e-12 ? clip(second, 0, 1) : null;
  }

  // Deterministic A* priority order: f score, path cost, grid x, then grid y.
  function lessNode(a, b) {
    return a.f < b.f || (a.f === b.f && (a.cost < b.cost
      || (a.cost === b.cost && (a.x < b.x || (a.x === b.x && a.y < b.y)))));
  }
  class MinHeap {
    constructor() { this.items = []; }
    push(item) {
      const items = this.items;
      let i = items.length;
      items.push(item);
      while (i > 0) {
        const p = (i - 1) >>> 1;
        if (!lessNode(item, items[p])) break;
        items[i] = items[p];
        i = p;
      }
      items[i] = item;
    }
    pop() {
      const items = this.items, first = items[0], last = items.pop();
      if (items.length) {
        let i = 0;
        while (true) {
          const left = 2 * i + 1;
          if (left >= items.length) break;
          const right = left + 1;
          const next = right < items.length && lessNode(items[right], items[left]) ? right : left;
          if (!lessNode(items[next], last)) break;
          items[i] = items[next];
          i = next;
        }
        items[i] = last;
      }
      return first;
    }
  }
  function roundEven(value) {
    const lower = Math.floor(value), fraction = value - lower;
    return fraction === 0.5 ? lower + (lower % 2) : Math.round(value);
  }

  class AUVEnv {
    constructor(config = {}) {
      if (!config || typeof config !== "object" || Array.isArray(config)) {
        throw new TypeError("config must be an object");
      }
      const settings = Object.assign({scenario: "reef", seed: 7, current_strength: 0.08,
        dt: 0.1, max_steps: 500, course_variety: true}, config);
      if (!Object.prototype.hasOwnProperty.call(MAPS, settings.scenario)) {
        throw new RangeError("scenario must be open, reef, or channel");
      }
      validateSeed(settings.seed);
      if (!Number.isFinite(settings.current_strength) || settings.current_strength < 0
          || settings.current_strength > 0.4) {
        throw new RangeError("current_strength must be finite and between 0 and 0.4 m/s");
      }
      if (!Number.isFinite(settings.dt) || settings.dt <= 0 || settings.dt > 10) {
        throw new RangeError("dt must be finite and in (0, 10] seconds");
      }
      if (!Number.isSafeInteger(settings.max_steps) || settings.max_steps < 1) {
        throw new RangeError("max_steps must be a positive safe integer");
      }
      if (typeof settings.course_variety !== "boolean") {
        throw new TypeError("course_variety must be a boolean");
      }
      this.config = Object.freeze(settings);
      this.rng = new SeededRNG(settings.seed);
      this._angles = new Float64Array(16);
      this._distances = new Float64Array(16);
      this._endX = new Float64Array(16);
      this._endY = new Float64Array(16);
      this.reset(settings.seed);
    }

    /** With no seed, continue this environment's RNG stream. */
    reset(seed, options) {
      if (seed !== undefined && seed !== null) validateSeed(seed);
      if (options !== undefined && options !== null
          && (typeof options !== "object" || Array.isArray(options))) {
        throw new TypeError("reset options must be an object");
      }
      let variety = this.config.course_variety;
      if (options && Object.prototype.hasOwnProperty.call(options, "course_variety")) {
        if (typeof options.course_variety !== "boolean") {
          throw new TypeError("course_variety reset option must be a boolean");
        }
        variety = options.course_variety;
      }
      if (seed !== undefined && seed !== null) this.rng = new SeededRNG(seed);
      this.position = [2, 3];
      this.goal = [17, 17];
      this.obstacles = MAPS[this.config.scenario].map(circle => circle.slice());
      if (variety) {
        for (let i = 0; i < 2; i++) this.position[i] += this.rng.uniform(-0.25, 0.25);
        for (let i = 0; i < 2; i++) this.goal[i] += this.rng.uniform(-0.35, 0.35);
        for (const obstacle of this.obstacles) {
          obstacle[0] += this.rng.uniform(-0.18, 0.18);
          obstacle[1] += this.rng.uniform(-0.18, 0.18);
        }
      }
      const jitter = variety ? this.rng.uniform(-0.12, 0.12) : 0;
      this.heading = wrapAngle(Math.atan2(this.goal[1] - this.position[1],
        this.goal[0] - this.position[0]) + jitter);
      this.speed = 0;
      const currentAngle = variety ? this.rng.uniform(-Math.PI, Math.PI) : 0;
      this.current = [this.config.current_strength * Math.cos(currentAngle),
        this.config.current_strength * Math.sin(currentAngle)];
      this.episode_step = 0;
      this.episode_return = 0;
      this.last_reward = 0;
      this.outcome = "running";
      this.reward_terms = emptyReward();
      this.trail = [this.position.slice()];
      this._heuristic_path = null;
      return {observation: this._observation(), info: this._info()};
    }

    _collisionXY(sx, sy, ex, ey, extraMargin = 0) {
      const radius = 0.25 + extraMargin, upperX = 20 - radius, upperY = 20 - radius;
      if (sx < radius - 1e-12 || sy < radius - 1e-12
          || sx > upperX + 1e-12 || sy > upperY + 1e-12) return 0;
      const dx = ex - sx, dy = ey - sy;
      let earliest = Infinity;
      if (dx > 0 && ex >= upperX) earliest = Math.min(earliest, (upperX - sx) / dx);
      else if (dx < 0 && ex <= radius) earliest = Math.min(earliest, (radius - sx) / dx);
      if (dy > 0 && ey >= upperY) earliest = Math.min(earliest, (upperY - sy) / dy);
      else if (dy < 0 && ey <= radius) earliest = Math.min(earliest, (radius - sy) / dy);
      for (let i = 0; i < this.obstacles.length; i++) {
        const obstacle = this.obstacles[i];
        const fraction = segmentCircleFraction(sx, sy, ex, ey,
          obstacle[0], obstacle[1], obstacle[2] + radius);
        if (fraction !== null) earliest = Math.min(earliest, fraction);
      }
      return earliest === Infinity ? null : clip(earliest, 0, 1);
    }

    _collision_fraction(start, end, extraMargin = 0) {
      return this._collisionXY(start[0], start[1], end[0], end[1], extraMargin);
    }

    // @trace-start physics
    step(action) {
      if (this.outcome !== "running") throw new Error("The episode is over; call reset before stepping again");
      if (!action || action.length !== 2 || !Number.isFinite(action[0]) || !Number.isFinite(action[1])) {
        throw new RangeError("action must have two finite numeric components");
      }
      const accel = clip(action[0], -1, 1), turn = clip(action[1], -1, 1);
      const px = this.position[0], py = this.position[1], dt = this.config.dt;
      const previousDistance = Math.hypot(this.goal[0] - px, this.goal[1] - py);
      const speed = clip(this.speed + dt * (0.9 * accel - 0.3 * this.speed), 0, 1.8);
      const heading = wrapAngle(this.heading + dt * 1.2 * turn);
      const ex = px + dt * (speed * Math.cos(heading) + this.current[0]);
      const ey = py + dt * (speed * Math.sin(heading) + this.current[1]);
      if (!Number.isFinite(ex) || !Number.isFinite(ey)) throw new RangeError("The proposed motion is not finite");
      const collision = this._collisionXY(px, py, ex, ey);
      const goal = segmentCircleFraction(px, py, ex, ey, this.goal[0], this.goal[1], 0.75);
      let fraction = 1;
      if (collision !== null && (goal === null || collision <= goal + 1e-12)) {
        fraction = collision;
        this.outcome = "collision";
      } else if (goal !== null) {
        fraction = goal;
        this.outcome = "success";
      }
      this.position = [clip(px + fraction * (ex - px), 0.25, 19.75),
        clip(py + fraction * (ey - py), 0.25, 19.75)];
      // Only position is clipped at contact; dynamics/costs retain the full dt.
      this.speed = speed;
      this.heading = heading;
      this.episode_step++;
      const terminated = this.outcome === "success" || this.outcome === "collision";
      const truncated = this.episode_step >= this.config.max_steps && !terminated;
      if (truncated) this.outcome = "timeout";
      const clearance = this._min_clearance(), elapsedScale = dt / 0.1;
      const near = Math.max(0, (0.75 - clearance) / 0.75);
      this.reward_terms = {
        progress: 6 * (previousDistance - Math.hypot(this.goal[0] - this.position[0], this.goal[1] - this.position[1])),
        goal_bonus: this.outcome === "success" ? 100 : 0,
        collision_penalty: this.outcome === "collision" ? -100 : 0,
        time_cost: -0.02 * elapsedScale,
        control_cost: -0.01 * elapsedScale * (accel * accel + 0.3 * turn * turn),
        proximity_cost: -0.12 * elapsedScale * near * near,
      };
      const terms = this.reward_terms;
      this.last_reward = terms.progress + terms.goal_bonus + terms.collision_penalty
        + terms.time_cost + terms.control_cost + terms.proximity_cost;
      this.episode_return += this.last_reward;
      this.trail.push(this.position.slice());
      if (this.trail.length > 600) this.trail.splice(0, this.trail.length - 600);
      return {observation: this._observation(), reward: this.last_reward,
        terminated, truncated, info: this._info()};
    }
    // @trace-end physics

    _min_clearance() {
      const x = this.position[0], y = this.position[1];
      let clearance = Math.min(x, y, 20 - x, 20 - y) - 0.25;
      for (let i = 0; i < this.obstacles.length; i++) {
        const obstacle = this.obstacles[i];
        clearance = Math.min(clearance, Math.hypot(obstacle[0] - x, obstacle[1] - y) - obstacle[2] - 0.25);
      }
      return clearance < 0 && clearance > -1e-9 ? 0 : clearance;
    }

    _sonar() {
      const x = this.position[0], y = this.position[1];
      for (let i = 0; i < 16; i++) {
        const angle = wrapAngle(this.heading + i * TAU / 16);
        const dx = Math.cos(angle), dy = Math.sin(angle);
        let distance = 6;
        if (dx > 1e-12) distance = Math.min(distance, Math.max(0, (20 - x) / dx));
        else if (dx < -1e-12) distance = Math.min(distance, Math.max(0, -x / dx));
        if (dy > 1e-12) distance = Math.min(distance, Math.max(0, (20 - y) / dy));
        else if (dy < -1e-12) distance = Math.min(distance, Math.max(0, -y / dy));
        for (let j = 0; j < this.obstacles.length; j++) {
          const obstacle = this.obstacles[j], rx = obstacle[0] - x, ry = obstacle[1] - y;
          const projection = dx * rx + dy * ry;
          const discriminant = obstacle[2] * obstacle[2] - (rx * rx + ry * ry - projection * projection);
          if (discriminant >= -1e-12) {
            const root = Math.sqrt(Math.max(discriminant, 0));
            if (projection + root >= 0) distance = Math.min(distance, Math.max(projection - root, 0));
          }
        }
        this._angles[i] = angle;
        this._distances[i] = distance;
        this._endX[i] = x + distance * dx;
        this._endY[i] = y + distance * dy;
      }
    }

    // @trace-start observation_features
    _observation() {
      this._sonar();
      const observation = new Float32Array(21);
      for (let i = 0; i < 16; i++) observation[i] = this._distances[i] / 6;
      const cosine = Math.cos(this.heading), sine = Math.sin(this.heading);
      const gx = this.goal[0] - this.position[0], gy = this.goal[1] - this.position[1];
      observation[16] = (cosine * gx + sine * gy) / DIAGONAL;
      observation[17] = (-sine * gx + cosine * gy) / DIAGONAL;
      observation[18] = this.speed / 1.8;
      observation[19] = (cosine * this.current[0] + sine * this.current[1]) / 0.4;
      observation[20] = (-sine * this.current[0] + cosine * this.current[1]) / 0.4;
      for (let i = 0; i < 21; i++) {
        if (!Number.isFinite(observation[i])) throw new RangeError("Environment state produced a non-finite observation");
        observation[i] = clip(observation[i], i < 16 || i === 18 ? 0 : -1, 1);
      }
      return observation;
    }
    // @trace-end observation_features

    _info() {
      return {outcome: this.outcome, reward_terms: Object.assign({}, this.reward_terms),
        distance_to_goal: Math.hypot(this.goal[0] - this.position[0], this.goal[1] - this.position[1]),
        min_clearance: this._min_clearance()};
    }

    snapshot() {
      this._sonar();
      const info = this._info(), sonar = new Array(16);
      for (let i = 0; i < 16; i++) {
        sonar[i] = {angle: this._angles[i], distance: this._distances[i],
          end_x: this._endX[i], end_y: this._endY[i]};
      }
      return {
        world: {width: 20, height: 20},
        vehicle: {x: this.position[0], y: this.position[1], heading: this.heading,
          speed: this.speed, radius: 0.25},
        goal: {x: this.goal[0], y: this.goal[1], radius: 0.75},
        obstacles: this.obstacles.map(circle => ({x: circle[0], y: circle[1], radius: circle[2]})),
        sonar,
        current: {x: this.current[0], y: this.current[1]},
        trail: this.trail.map(point => Array.from(point)),
        episode_step: this.episode_step, episode_return: this.episode_return,
        last_reward: this.last_reward, distance_to_goal: info.distance_to_goal,
        min_clearance: info.min_clearance, outcome: this.outcome,
        reward_terms: info.reward_terms,
      };
    }

    _plan_heuristic_path() {
      const resolution = 0.5, count = 41, size = count * count;
      const sx = roundEven(this.position[0] / resolution), sy = roundEven(this.position[1] / resolution);
      const gx = roundEven(this.goal[0] / resolution), gy = roundEven(this.goal[1] / resolution);
      const start = sx * count + sy, goal = gx * count + gy;
      const costs = new Float64Array(size); costs.fill(Infinity);
      const parents = new Int32Array(size); parents.fill(-1);
      const frontier = new MinHeap();
      frontier.push({f: Math.hypot(sx - gx, sy - gy), cost: 0, x: sx, y: sy});
      costs[start] = 0;
      const offsets = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]];
      let found = false;
      while (frontier.items.length) {
        const node = frontier.pop(), id = node.x * count + node.y;
        if (node.cost > costs[id] + 1e-12) continue;
        if (id === goal) { found = true; break; }
        for (let i = 0; i < offsets.length; i++) {
          const dx = offsets[i][0], dy = offsets[i][1], nx = node.x + dx, ny = node.y + dy;
          if (nx < 0 || ny < 0 || nx >= count || ny >= count) continue;
          if (this._collisionXY(node.x * resolution, node.y * resolution,
            nx * resolution, ny * resolution, 0.4) !== null) continue;
          const neighbor = nx * count + ny, candidate = node.cost + Math.hypot(dx, dy);
          if (candidate < costs[neighbor]) {
            costs[neighbor] = candidate;
            parents[neighbor] = id;
            frontier.push({f: candidate + Math.hypot(nx - gx, ny - gy),
              cost: candidate, x: nx, y: ny});
          }
        }
      }
      if (!found) return [Array.from(this.goal)];
      const route = [];
      let node = goal;
      while (true) {
        route.push([Math.floor(node / count) * resolution, (node % count) * resolution]);
        if (node === start) break;
        node = parents[node];
      }
      route.reverse();
      route.push(Array.from(this.goal));
      return route;
    }

    /** Full-map A* steering demo; never called by step or a learned actor. */
    heuristicAction() {
      if (this.outcome !== "running") return new Float32Array(2);
      if (this._heuristic_path === null) this._heuristic_path = this._plan_heuristic_path();
      const path = this._heuristic_path, px = this.position[0], py = this.position[1];
      while (path.length > 1 && Math.hypot(path[0][0] - px, path[0][1] - py) < 0.65) path.shift();
      let target = path[0];
      for (let i = 1; i < path.length; i++) {
        const waypoint = path[i];
        if (Math.hypot(waypoint[0] - px, waypoint[1] - py) > 2.5) break;
        if (this._collisionXY(px, py, waypoint[0], waypoint[1], 0.28) !== null) break;
        target = waypoint;
      }
      let dx = target[0] - px, dy = target[1] - py, length = Math.hypot(dx, dy);
      if (length < 1e-9) {
        dx = this.goal[0] - px; dy = this.goal[1] - py;
        length = Math.max(Math.hypot(dx, dy), 1e-9);
      }
      const vx = 1.2 * dx / length - this.current[0], vy = 1.2 * dy / length - this.current[1];
      const error = wrapAngle(Math.atan2(vy, vx) - this.heading);
      const targetSpeed = Math.min(1.5, Math.hypot(vx, vy)) * Math.max(0.15, Math.cos(error));
      const acceleration = (2 * (targetSpeed - this.speed) + 0.3 * this.speed) / 0.9;
      const yaw = 2 * error / 1.2;
      return new Float32Array([clip(acceleration, -1, 1), clip(yaw, -1, 1)]);
    }
  }

  Object.assign(AUVEnv, CONSTANTS);
  Object.assign(AUVEnv.prototype, CONSTANTS);
  const exports = {AUVEnv, SeededRNG};
  root.AUVEnvironment = exports;
  if (typeof module !== "undefined" && module.exports) module.exports = exports;
})(typeof globalThis !== "undefined" ? globalThis : self);
