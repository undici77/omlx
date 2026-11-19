const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');

const source = fs.readFileSync(
    path.join(__dirname, '../omlx_web/static/js/dashboard.js'), 'utf8'
);
const context = {
    localStorage: {getItem: () => null},
    THEME_STORAGE_KEY: 'theme',
    ENHANCED_READABILITY_KEY: 'readability',
    window: {t: key => key},
    navigator: {language: 'en'},
    document: {},
};
const create = vm.runInNewContext(source + '\n dashboard;', context);
const state = create();

// Shown only on the validated architecture.
const flash = {id: 'flash', config_model_type: 'qwen4_exp'};
assert.equal(state.buildModelSettingsState(flash, {}).steering_supported, true);
assert.equal(state.buildModelSettingsState({config_model_type: 'Qwen4-Exp'}, {}).steering_supported, true);
assert.equal(state.buildModelSettingsState({config_model_type: 'qwen3_5'}, {}).steering_supported, false);

// Saved settings load into one strength field.
const loaded = state.buildModelSettingsState(flash, {
    steering_pack: 'terse', steering_attn_scale: -1, steering_scope: 'prefill+k',
});
assert.equal(loaded.steering_pack, 'terse');
assert.equal(loaded.steering_scale, -1);
assert.equal(loaded.steering_scope, 'prefill+k');

// The pack's component decides which scale setting is saved.
state.steeringPacks = [
    {name: 'terse', component: 'attn', recommended: {attn: -1}},
    {name: 'ffnpack', component: 'ffn', recommended: null},
];
state.modelSettings = {...loaded};
assert.equal(state.steeringRecommendedScale(), -1);
assert.deepEqual(JSON.parse(JSON.stringify(state.steeringPayload(state.modelSettings))), {
    steering_pack: 'terse',
    steering_attn_scale: -1,
    steering_ffn_scale: null,
    steering_scope: 'prefill+k',
});
state.modelSettings = {...loaded, steering_pack: 'ffnpack', steering_scale: 0.5, steering_scope: ''};
assert.equal(state.steeringRecommendedScale(), null);
assert.deepEqual(JSON.parse(JSON.stringify(state.steeringPayload(state.modelSettings))), {
    steering_pack: 'ffnpack',
    steering_attn_scale: null,
    steering_ffn_scale: 0.5,
    steering_scope: null,
});

// Off clears the pack (the server then drops every dependent field).
state.modelSettings = {...loaded, steering_pack: ''};
assert.deepEqual(JSON.parse(JSON.stringify(state.steeringPayload(state.modelSettings))), {steering_pack: null});

console.log('steering_ui ok');
