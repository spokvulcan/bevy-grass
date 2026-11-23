//! bevy_grass::shaders::grass_prepass — grass in Bevy's prepass and in its
//! deferred G-buffer: depth always; normals, motion vectors and the G-buffer
//! when the view asks for them. Group 0 is Bevy's prepass view group, 1 is
//! its empty group and 2 is the grass group, as in the other grass draws.
//! What Bevy's prepass expects of its outputs is in `bevy_bridge`.
//!
//! Motion vectors are exact for the levels of detail below
//! `GrassSettings::motion_lods`: the blade is built a second time as it stood
//! last frame, with last frame's bend, clock and camera. A blade of any other
//! level is built once and taken to have stood still, so its motion vector
//! is the camera's alone. A game that moved its origin this frame says how in
//! `GrassMoved`.

import package::shaders::bevy_bridge::{
    view_world_from_view, view_clip_from_view, view_clip_from_world, view_viewport_width,
    view_is_orthographic, prepass_normal,
};
import package::shaders::blade::{blade_ref, blade_vertex, current_eye, previous_eye, fields, grass_view};

@if(MOTION_VECTOR_PREPASS)
import package::shaders::bevy_bridge::{
    previous_view_from_world, previous_clip_from_view, previous_clip_from_world, prepass_motion_vector,
};

@if(DEFERRED_PREPASS)
import package::shaders::look::{blade_look, look_terms, unpack_look, normal_spread, blade_surface, transmitted};
@if(DEFERRED_PREPASS)
import package::shaders::bevy_bridge::{grass_pbr_input, grass_gbuffer, DEFERRED_LIGHTING_PASS_ID};

struct VertexOutput {
    /// Invariant, so the depth written here is bit for bit the main pass's
    /// (see `grass.wesl`).
    @builtin(position) @invariant position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @if(NORMAL_PREPASS_OR_DEFERRED_PREPASS)
    @location(1) world_normal: vec3<f32>,
    @if(MOTION_VECTOR_PREPASS)
    @location(2) previous_world_position: vec4<f32>,
    /// The blade's `Look`, packed by `look.wesl`.
    @if(DEFERRED_PREPASS)
    @location(3) look_color: vec4<f32>,
    @if(DEFERRED_PREPASS)
    @location(4) look_terms: vec4<f32>,
}

@if(PREPASS_FRAGMENT)
struct FragmentOutput {
    @if(NORMAL_PREPASS)
    @location(0) normal: vec4<f32>,
    @if(MOTION_VECTOR_PREPASS)
    @location(1) motion_vector: vec2<f32>,
    @if(DEFERRED_PREPASS)
    @location(2) deferred: vec4<u32>,
    @if(DEFERRED_PREPASS)
    @location(3) deferred_lighting_pass_id: u32,
}

@vertex
fn vertex(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let blade_at = blade_ref(vertex_index, instance_index);
    let blade = blade_vertex(
        blade_at,
        current_eye(view_world_from_view(), view_clip_from_view(), view_clip_from_world(), view_viewport_width()),
    );
    var out: VertexOutput;
    out.world_position = vec4<f32>(blade.world_position, 1.0);
    out.position = view_clip_from_world() * out.world_position;

@if(NORMAL_PREPASS_OR_DEFERRED_PREPASS)
    out.world_normal = blade.normal;

@if(MOTION_VECTOR_PREPASS) {
    // Last frame's camera is carried into this frame's world, where the blade
    // stands, and the blade as it stood back into last frame's, where Bevy
    // asks that camera. Both are the identity but when the origin moved.
    let moved = grass_view.previous_from_world;
    var stood = blade.world_position;
    if blade_at.lod < grass_view.motion_lods {
        stood = blade_vertex(
            blade_at,
            previous_eye(
                previous_view_from_world() * moved,
                previous_clip_from_view(),
                previous_clip_from_world() * moved,
                view_viewport_width(),
            ),
        ).world_position;
    }
    out.previous_world_position = moved * vec4<f32>(stood, 1.0);
}

@if(DEFERRED_PREPASS) {
    let look = blade_look(blade_at.instance, fields[blade_at.instance.field], blade.t);
    out.look_color = look.color;
    out.look_terms = look_terms(look);
}
    return out;
}

@if(PREPASS_FRAGMENT)
@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var out: FragmentOutput;

@if(NORMAL_PREPASS_OR_DEFERRED_PREPASS) {
    // Blades are two-sided: the back face's normal is flipped.
    let normal = normalize(in.world_normal) * select(-1.0, 1.0, is_front);
@if(NORMAL_PREPASS)
    out.normal = prepass_normal(normal);

@if(DEFERRED_PREPASS) {
    // Translucency, as near the forward pass as a G-buffer allows: a face
    // turned away from the sun is stored as its sunward side, with its colour
    // dimmed and tinted by the light that passes through it. The forward pass
    // keeps such a face's ambient light whole; here it is dimmed too.
    let sun = grass_view.sun;
    let backlit = sun.w > 0.5 && dot(normal, sun.xyz) < 0.0;
    let lit_side = select(normal, -normal, backlit);
    var look = unpack_look(in.look_color, in.look_terms);
    look.color = vec4<f32>(look.color.rgb * select(vec3<f32>(1.0), transmitted(look.color), backlit), look.color.a);
    out.deferred = grass_gbuffer(grass_pbr_input(
        blade_surface(
            look,
            lit_side,
            normal_spread(in.world_normal),
            in.position,
            in.world_position,
            view_is_orthographic(),
        ),
    ));
    out.deferred_lighting_pass_id = DEFERRED_LIGHTING_PASS_ID;
}
}

@if(MOTION_VECTOR_PREPASS)
    out.motion_vector = prepass_motion_vector(in.world_position, in.previous_world_position);
    return out;
}
